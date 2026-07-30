"""SageMaker serving entry point: turn a raw Avazu row into a pCTR score.

The counterpart to train.py. SageMaker's SKLearn container imports this module
and calls four handlers per request:

    model_fn(model_dir)              once, at container start
    input_fn(body, content_type)     per request -- deserialize
    predict_fn(data, model)          per request -- score
    output_fn(prediction, accept)    per request -- serialize

## The one thing this file must never do

**Return anything but a pCTR score.** No bids, no budgets, no auction rules, no
ranking. The model is bid-agnostic by contract (CLAUDE.md), and ranking lives in
rank_candidates.py, which plays the calling auction service. Keeping that
boundary sharp is why an ads org can retrain the pCTR model without touching
auction logic, and change auction logic without retraining.

## Feature parity

This module does NOT reimplement feature engineering. It imports `features` --
the identical module train.py used, shipped into the container by `source_dir`.
Any "small fix" applied here and not there reintroduces exactly the
training/serving skew the shared module exists to prevent.

`model_fn` additionally calls `features.assert_config_matches()`, so a mismatch
between the saved config and this code crashes the container at startup instead
of silently scoring a feature space the model never saw. A dead endpoint is
vastly preferable to one quietly returning wrong pCTRs into an auction.

## Batch Transform

CLAUDE.md requires Phase 5 to reuse this file unchanged, so `input_fn` handles
text/csv (with a header row) as well as application/json. Nothing here is
endpoint-specific.
"""

from __future__ import annotations

import io
import json
import os
import sys

import joblib
import pandas as pd

# The container flattens source_dir onto /opt/ml/code; this also lets the module
# be imported directly for local testing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402

JSON_TYPE = "application/json"
CSV_TYPE = "text/csv"


def model_fn(model_dir: str) -> dict:
    """Load the model and its featurizer, and refuse to start if they disagree."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    featurizer = joblib.load(os.path.join(model_dir, "featurizer.joblib"))

    with open(os.path.join(model_dir, "feature_config.json")) as fh:
        config = json.load(fh)

    # Fails loudly on training/serving skew. See module docstring.
    features.assert_config_matches(config)

    print(f"loaded model {type(model).__name__}, "
          f"featurizer n_features={featurizer.n_features}, config v{config['version']}")
    return {"model": model, "featurizer": featurizer, "config": config}


def input_fn(request_body, content_type: str = JSON_TYPE) -> pd.DataFrame:
    """Raw request body -> DataFrame of raw Avazu columns.

    Note what is NOT done here: no encoding, no hashing, no column derivation.
    This only deserializes. All feature construction happens in predict_fn via
    the shared `features` module, so there is exactly one code path from raw
    column to feature vector.
    """
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    # Strip charset suffixes such as "application/json; charset=utf-8".
    base_type = (content_type or JSON_TYPE).split(";")[0].strip()

    if base_type == JSON_TYPE:
        payload = json.loads(request_body)
        # Accept a single record or a batch; a bare dict is the common case when
        # hand-testing with one row.
        records = payload if isinstance(payload, list) else [payload]
        # dtype=str throughout, matching how train.py read the CSVs. Letting
        # pandas infer here would turn "1" into 1 and produce the token
        # "banner_pos=1" vs "banner_pos=1.0" -- a different hash bucket, and
        # silent skew.
        return pd.DataFrame(records).astype(str)

    if base_type == CSV_TYPE:
        # Header row required: the model is addressed by column name, not
        # position. Positional CSV would make column order a hidden contract
        # between caller and model.
        return pd.read_csv(io.StringIO(request_body), dtype=str)

    raise ValueError(
        f"Unsupported content type {content_type!r}. Use {JSON_TYPE} or "
        f"{CSV_TYPE} (with a header row)."
    )


def predict_fn(input_df: pd.DataFrame, model_artifacts: dict) -> list[float]:
    """Score rows. Returns P(click) per row -- and nothing else."""
    model = model_artifacts["model"]
    featurizer = model_artifacts["featurizer"]

    # A caller may hand back a row straight from test.csv, label column and all.
    # Dropping it is a convenience; it is never a feature.
    df = input_df.drop(columns=[features.LABEL], errors="ignore")

    X = features.transform(df, featurizer)
    return model.predict_proba(X)[:, 1].tolist()


def output_fn(prediction: list[float], accept: str = JSON_TYPE) -> tuple[str, str]:
    """Serialize scores. One float per input row, order preserved."""
    base_type = (accept or JSON_TYPE).split(";")[0].strip()

    if base_type == CSV_TYPE:
        return "\n".join(str(p) for p in prediction) + "\n", CSV_TYPE
    if base_type in (JSON_TYPE, "*/*"):
        return json.dumps(prediction), JSON_TYPE

    raise ValueError(f"Unsupported accept type {accept!r}")
