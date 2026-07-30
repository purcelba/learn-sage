"""SageMaker training entry point: fit a pCTR model on Avazu.

## What SageMaker does vs. what this script owns

SageMaker provisions an instance, pulls the SKLearn framework container, copies
this directory into it, downloads the S3 input channels to local paths, runs
this file as `python train.py --arg ...`, then tars whatever landed in
`/opt/ml/model` and uploads it to S3. It does none of the machine learning.

This script owns all of it: which columns are features, how they're encoded,
what model is fit, and how it's evaluated. That split is the point -- the
platform runs the job, the team owns what runs. Same division as LyftLearn
Compute.

The paths below are the interface between the two. SageMaker sets them as
environment variables inside the container; the argparse defaults read those,
which is also what makes this script runnable locally with explicit paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# Works both inside the container (where SageMaker flattens source_dir onto the
# path) and locally when run as `python src/train.py`.
sys.path.insert(0, str(Path(__file__).parent))
import features  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Hyperparameters. SageMaker passes anything in the estimator's
    # `hyperparameters` dict as `--key value` CLI args -- there's no magic here,
    # just a subprocess invocation.
    ap.add_argument("--n-features", type=int, default=features.N_FEATURES)
    ap.add_argument("--C", type=float, default=1.0,
                    help="inverse L2 regularization strength; smaller = stronger")
    ap.add_argument("--max-iter", type=int, default=200)

    # SageMaker-provided paths, defaulted from the container's env vars.
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    ap.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    return ap.parse_args()


def log_environment() -> None:
    """Record the container's library versions in the training log.

    Not decoration. The model artifact is a pickle, and pickles are sensitive to
    the versions that wrote them -- particularly across the numpy 1.x/2.x
    boundary. Phase 4 has to load this artifact and reproduce the endpoint's
    prediction locally; when that comparison fails, the first question is
    "were these the same libraries?" Logging it here makes that answerable
    instead of a guess.
    """
    import sklearn
    import scipy
    print("--- container environment ---")
    print(f"python      {sys.version.split()[0]}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"numpy        {np.__version__}")
    print(f"scipy        {scipy.__version__}")
    print(f"pandas       {pd.__version__}")
    print(f"joblib       {joblib.__version__}")
    print("-----------------------------")


def load_split(channel_dir: str, filename: str) -> pd.DataFrame:
    path = Path(channel_dir) / filename
    if not path.exists():
        # A channel directory that exists but is empty is the classic symptom of
        # an S3 prefix typo, so show what actually arrived.
        contents = list(Path(channel_dir).glob("*")) if Path(channel_dir).exists() else []
        raise SystemExit(f"Expected {path}. Channel contains: {contents}")
    # dtype=str for the same reason prepare_data.py used it: these are
    # categorical hex strings and IDs, not numbers. Inferring dtypes here would
    # produce different tokens than serving does -- skew introduced by a default.
    return pd.read_csv(path, dtype=str)


def main() -> None:
    args = parse_args()
    log_environment()

    train_df = load_split(args.train, "train.csv")
    test_df = load_split(args.test, "test.csv")
    print(f"train={len(train_df):,} rows   test={len(test_df):,} rows")

    y_train = train_df[features.LABEL].astype(int).to_numpy()
    y_test = test_df[features.LABEL].astype(int).to_numpy()
    print(f"click rate: train={y_train.mean():.4f}  test={y_test.mean():.4f}")

    featurizer = features.build_featurizer(n_features=args.n_features)
    X_train = features.transform(train_df, featurizer)
    X_test = features.transform(test_df, featurizer)
    print(f"feature matrix: {X_train.shape}, "
          f"density={X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.6f}")

    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="lbfgs",      # handles sparse input and scales to 2**18 columns
        penalty="l2",
    )
    model.fit(X_train, y_train)
    if hasattr(model, "n_iter_") and model.n_iter_[0] >= args.max_iter:
        # Silent non-convergence would quietly cost AUC, so say so.
        print(f"WARNING: hit max_iter={args.max_iter} without converging. "
              f"Raise --max-iter or lower --C.")

    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    ll = log_loss(y_test, p_test)

    # Baseline: predict the training base rate for every row. A model that can't
    # beat this has learned nothing usable, and it's the honest reference point
    # for log loss -- unlike AUC, log loss has no fixed "no better than chance"
    # value to compare against.
    baseline = np.full_like(p_test, y_train.mean())
    ll_baseline = log_loss(y_test, baseline)

    print(f"\nvalidation AUC      : {auc:.4f}")
    print(f"validation log loss : {ll:.4f}")
    print(f"baseline log loss   : {ll_baseline:.4f}  (predict base rate)")
    print(f"log loss improvement: {(ll_baseline - ll) / ll_baseline:.1%}")

    # Calibration matters for a pCTR model specifically: Phase 4 ranks by
    # bid x pCTR, so a systematically inflated score distorts the auction even
    # when AUC -- which only sees ordering -- looks fine.
    print(f"mean predicted pCTR : {p_test.mean():.4f}  (actual {y_test.mean():.4f})")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(featurizer, model_dir / "featurizer.joblib")

    # The featurizer is stateless, so this config is the part that actually
    # carries the parity contract. inference.py asserts against it at load time.
    config = dict(features.FEATURE_CONFIG, n_features=args.n_features)
    (model_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    # An eval report next to the model is the small version of "versioned
    # artifact gated by an eval report" -- the Phase 3 registry idea.
    (model_dir / "metrics.json").write_text(json.dumps({
        "roc_auc": auc,
        "log_loss": ll,
        "baseline_log_loss": ll_baseline,
        "mean_predicted_ctr": float(p_test.mean()),
        "actual_ctr": float(y_test.mean()),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "hyperparameters": {"C": args.C, "n_features": args.n_features,
                            "max_iter": args.max_iter},
    }, indent=2))

    print(f"\nwrote artifacts to {model_dir}: "
          f"{sorted(p.name for p in model_dir.iterdir())}")


if __name__ == "__main__":
    main()
