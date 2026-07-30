"""Phase 4 acceptance criteria 1 and 2.

Criterion 1: invoke_endpoint returns a probability (0-1) that moves sensibly
             when an input feature you'd expect to matter is varied.
Criterion 2: the endpoint's prediction matches scoring the same row locally
             through model + featurizer -- training/serving parity, not merely
             "an endpoint responds."

Criterion 2 is the one that carries weight. An endpoint returning 0.34 looks
identical whether or not it agrees with training; only comparing against a local
score can tell the difference. This is the small-scale version of the parity
guarantee a feature store exists to provide.

Usage:
    python invoke_endpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3
import joblib
import pandas as pd

sys.path.insert(0, "src")
import features  # noqa: E402

REGION = "us-east-1"
ENDPOINT_NAME = "learn-sage-pctr"
LOCAL_ARTIFACT = Path("/tmp/artifact")

# Tolerance for the parity comparison. The endpoint and this process run
# different Python versions (3.10 vs 3.12) and different numpy builds (2.1 vs
# 2.5), so bit-identical output isn't guaranteed; but the same model on the same
# features should agree far more tightly than this.
TOLERANCE = 1e-9


def invoke(runtime, record: dict) -> float:
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(record),
    )
    return json.loads(resp["Body"].read())[0]


def main() -> None:
    session = boto3.Session(region_name=REGION)
    runtime = session.client("sagemaker-runtime")

    test_df = pd.read_csv("data/test.csv", dtype=str)
    base = test_df.iloc[[0]].drop(columns=["click"]).to_dict(orient="records")[0]

    print("=" * 62)
    print("CRITERION 1: probability-shaped, and responsive to inputs")
    print("=" * 62)

    p_base = invoke(runtime, base)
    print(f"  baseline row               -> pCTR = {p_base:.6f}")
    assert 0.0 < p_base < 1.0, f"not a probability: {p_base}"
    print(f"  in (0, 1): PASS")

    # banner_pos is an ad-side placement feature -- position on the page is one
    # of the strongest known CTR drivers in display advertising, so if anything
    # moves the score, this should.
    print(f"\n  varying banner_pos (baseline is {base['banner_pos']!r}):")
    moved = False
    for bp in ["0", "1", "2", "3", "4", "5", "7"]:
        variant = dict(base, banner_pos=bp)
        p = invoke(runtime, variant)
        delta = p - p_base
        flag = "  <- baseline" if bp == base["banner_pos"] else ""
        print(f"    banner_pos={bp} -> pCTR = {p:.6f}  (delta {delta:+.6f}){flag}")
        if abs(delta) > 1e-6:
            moved = True
    print(f"  score responds to banner_pos: {'PASS' if moved else 'FAIL'}")

    print()
    print("=" * 62)
    print("CRITERION 2: endpoint matches local scoring (training/serving parity)")
    print("=" * 62)

    model = joblib.load(LOCAL_ARTIFACT / "model.joblib")
    featurizer = joblib.load(LOCAL_ARTIFACT / "featurizer.joblib")
    with open(LOCAL_ARTIFACT / "feature_config.json") as fh:
        features.assert_config_matches(json.load(fh))

    rows = test_df.head(10).drop(columns=["click"])
    local_scores = model.predict_proba(features.transform(rows, featurizer))[:, 1]

    print(f"  {'row':>4}  {'endpoint':>12}  {'local':>12}  {'abs diff':>12}")
    worst = 0.0
    for i, (_, row) in enumerate(rows.iterrows()):
        p_endpoint = invoke(runtime, row.to_dict())
        p_local = float(local_scores[i])
        diff = abs(p_endpoint - p_local)
        worst = max(worst, diff)
        print(f"  {i:>4}  {p_endpoint:>12.9f}  {p_local:>12.9f}  {diff:>12.2e}")

    print(f"\n  max abs difference: {worst:.2e}  (tolerance {TOLERANCE:.0e})")
    if worst <= TOLERANCE:
        print("  PARITY: PASS -- endpoint and local scoring agree")
    else:
        print("  PARITY: FAIL -- serving diverges from training")
        sys.exit(1)


if __name__ == "__main__":
    main()
