"""Raw Avazu columns -> one flat numeric vector.

**This module is the single implementation of "raw column becomes feature."**
`train.py` imports it now; `inference.py` will import it unchanged in Phase 4.
Neither may reimplement any of this. If training and serving ever disagree about
how a column becomes a feature, the model is scoring inputs it was never trained
on and the failure is silent -- no error, just quietly wrong predictions. That's
the training/serving skew a feature store exists to prevent, and at this scale
the whole defense is "there is only one copy of this code."

## Why hashing rather than one-hot

Every feature becomes a `"column=value"` token, and all tokens go through one
`FeatureHasher` into a single flat vector. Measured cardinalities on the 40k-row
training split:

    device_ip      33,635 distinct   (84% of rows are unique!)
    device_id       6,926
    device_model    2,360
    C14             1,426
    site_id         1,053
    ...             ~48,000 distinct tokens total

One-hot needs a fitted vocabulary, so a value never seen in training has no
column to land in. `site_id` is unbounded in production -- new sites appear
constantly -- and each new one would need a retrain before the model could
represent it at all. A hash function has no vocabulary: an unseen `site_id`
lands in some bucket immediately and the model degrades gracefully instead of
erroring. This is why production ads systems hash.

It also satisfies the contract's requirement that `device_id`/`device_ip` be
hashed into a bounded space rather than one-hot encoded. They aren't behavioral
features -- `device_ip` is nearly a row identifier here -- and one-hot encoding
them would hand the model 33k columns to memorize. In a real system these would
be replaced by behavioral aggregates from a feature store, not used raw.

## The skew risk hashing does NOT remove

`FeatureHasher` is **stateless** -- it fits nothing. So there's no vocabulary to
drift, which is exactly the appeal. But that relocates the risk rather than
eliminating it: parity now depends on this file and on `N_FEATURES` being
identical at training and serving time. A silent change to `token_rows()` or to
the hash size breaks parity just as thoroughly as a stale vocabulary would, and
just as invisibly.

Hence `FEATURE_CONFIG`, persisted alongside the model and asserted at load time.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction import FeatureHasher

LABEL = "click"

# Dropped, not encoded. `hour` arrives as YYMMDDHH -- all 240 distinct values in
# our sample are specific calendar dates in October 2014. Encoding that as a
# category would memorize dates that can never recur; the model would carry
# 240 features of pure noise. What's actually predictive is the *cyclical*
# position: time of day, and weekday vs weekend. So we derive those and throw
# the timestamp away.
RAW_DROP = ["hour"]

DERIVED = ["hour_of_day", "day_of_week"]

# Everything else, grouped as in PHASES.md. The grouping is documentation, not
# mechanism -- all groups concatenate into one flat vector. There is no separate
# "user model" and "ad model"; a single LogisticRegression sees one vector.
CONTEXT = ["C1"]
AD = [
    "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "C15", "C16",
]
USER_DEVICE = [
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
]
ANONYMIZED = ["C14", "C17", "C18", "C19", "C20", "C21"]

PASSTHROUGH = CONTEXT + AD + USER_DEVICE + ANONYMIZED
FEATURE_COLUMNS = DERIVED + PASSTHROUGH

# 2**18 = 262,144 slots against ~48,000 distinct tokens: a load factor near
# 0.18, so hash collisions are rare. Too small and unrelated features share a
# weight (silently, and unfixably); too large just wastes memory on a sparse
# matrix. This is a hyperparameter -- but it is part of the parity contract, so
# it can only be changed by retraining, never on the serving side alone.
N_FEATURES = 2 ** 18

FEATURE_CONFIG = {
    "version": 1,
    "n_features": N_FEATURES,
    "feature_columns": FEATURE_COLUMNS,
    "label": LABEL,
}


def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """YYMMDDHH -> hour_of_day + day_of_week; drops the raw timestamp."""
    out = df.copy()
    ts = pd.to_datetime(out["hour"], format="%y%m%d%H")
    out["hour_of_day"] = ts.dt.hour.astype(str)
    out["day_of_week"] = ts.dt.dayofweek.astype(str)
    return out.drop(columns=RAW_DROP)


def token_rows(df: pd.DataFrame) -> list[list[str]]:
    """One list of "column=value" tokens per row.

    Prefixing with the column name is what keeps feature spaces separate: a bare
    value of "1" from `banner_pos` and from `device_type` would otherwise hash to
    the same bucket and share a weight, silently conflating two unrelated
    features.
    """
    frame = derive_time_features(df)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    frame = frame[FEATURE_COLUMNS].astype(str)
    cols = frame.columns.to_list()
    return [
        [f"{col}={val}" for col, val in zip(cols, row)]
        for row in frame.itertuples(index=False, name=None)
    ]


def build_featurizer(n_features: int = N_FEATURES) -> FeatureHasher:
    """The encoder. Stateless, but persisted so serving can't drift from it."""
    return FeatureHasher(n_features=n_features, input_type="string", alternate_sign=True)


def transform(df: pd.DataFrame, featurizer: FeatureHasher):
    """Raw dataframe -> sparse feature matrix. The only path to a feature vector."""
    return featurizer.transform(token_rows(df))


def assert_config_matches(loaded_config: dict) -> None:
    """Fail loudly if serving-time feature config differs from training time.

    Called by `inference.py` in Phase 4. Without it, a change to this file
    between training and deployment produces no error at all -- just a model
    quietly scoring a feature space it never saw. An explicit crash at load time
    is enormously preferable to silently wrong pCTR values feeding an auction.
    """
    for key in ("version", "n_features", "feature_columns"):
        if loaded_config.get(key) != FEATURE_CONFIG[key]:
            raise RuntimeError(
                f"Feature config mismatch on '{key}': the saved model was trained with "
                f"{loaded_config.get(key)!r} but this code produces "
                f"{FEATURE_CONFIG[key]!r}. Training/serving skew -- retrain, or check "
                f"out the features.py that matches this model artifact."
            )
