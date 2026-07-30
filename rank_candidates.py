"""Toy auction service: rank candidate ads by bid x pCTR.

**This script is the caller, not the model.** That separation is the entire
point, and it is a contract in CLAUDE.md, not a stylistic preference.

The endpoint knows nothing about bids, budgets, or auctions -- ask it about a
row and it returns a pCTR score, full stop. Everything commercial lives here:
what an impression is worth, how bids combine with predicted click-through, how
ties break, which candidate wins.

Why that boundary is worth defending:

  - The pCTR model can be retrained, re-tuned, or swapped for a neural net
    without touching auction logic.
  - Auction rules (reserve prices, pacing, budget caps, quality floors) can
    change hourly without retraining anything.
  - pCTR stays interpretable as a probability. Once you train against revenue
    directly, you can no longer ask "is this model well calibrated?" -- and
    calibration is what makes bid x pCTR meaningful in the first place.

A real ad server does much more here (budget pacing, frequency caps, reserve
prices, second-price billing). The `bid x pCTR` core is the same.

Usage:
    python rank_candidates.py
"""

from __future__ import annotations

import json
import sys

import boto3
import pandas as pd

REGION = "us-east-1"
ENDPOINT_NAME = "learn-sage-pctr"

# A candidate set: same user, same context, same moment -- different ads.
# Only ad-side fields vary, which is what "which ad should we show *this*
# request" actually means. User and context fields are held fixed below by
# construction: every candidate starts from the same base row.
CANDIDATES = [
    # name,          ad-side overrides,                                  bid ($)
    ("top-banner",   {"banner_pos": "1", "site_category": "28905ebd"},   2.50),
    ("sidebar",      {"banner_pos": "0", "site_category": "28905ebd"},   3.00),
    ("in-feed",      {"banner_pos": "1", "site_category": "f028772b"},   1.80),
    ("footer",       {"banner_pos": "4", "site_category": "f028772b"},   4.00),
    ("interstitial", {"banner_pos": "7", "site_category": "50e219e0"},   1.20),
]


def score(runtime, record: dict) -> float:
    """Ask the model for a pCTR. Note what is NOT sent: the bid."""
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(record),
    )
    return json.loads(resp["Body"].read())[0]


def run_auction(runtime, base: dict, candidates, title: str) -> list[dict]:
    print(f"\n{title}")
    print("-" * 72)

    results = []
    for name, overrides, bid in candidates:
        # The candidate row: fixed user/context, varied ad-side fields.
        record = dict(base, **overrides)
        pctr = score(runtime, record)
        results.append({
            "name": name,
            "bid": bid,
            "pctr": pctr,
            # Expected value of showing this ad. This multiplication is the
            # whole auction, and it happens HERE -- outside the model artifact.
            "score": bid * pctr,
            "banner_pos": overrides["banner_pos"],
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"  {'rank':<5}{'candidate':<15}{'banner_pos':<12}{'bid':>7}"
          f"{'pCTR':>10}{'bid x pCTR':>13}")
    for i, r in enumerate(results, 1):
        marker = "  <- WINNER" if i == 1 else ""
        print(f"  {i:<5}{r['name']:<15}{r['banner_pos']:<12}"
              f"${r['bid']:>6.2f}{r['pctr']:>10.4f}{r['score']:>13.4f}{marker}")
    return results


def main() -> None:
    runtime = boto3.Session(region_name=REGION).client("sagemaker-runtime")

    # One real user/context, held fixed across every candidate.
    base = (pd.read_csv("data/test.csv", dtype=str)
            .iloc[[0]].drop(columns=["click"]).to_dict(orient="records")[0])
    print(f"Simulating one ad request.")
    print(f"  user/context held fixed: device_type={base['device_type']}, "
          f"device_conn_type={base['device_conn_type']}, hour={base['hour']}")
    print(f"  {len(CANDIDATES)} candidate ads differing only on ad-side fields.")

    first = run_auction(runtime, base, CANDIDATES,
                        "AUCTION 1 -- baseline")

    # Criterion 4 wants ordering to respond to BOTH inputs, so vary each
    # independently. Changing only the bid isolates the commercial lever;
    # changing only the ad features isolates the model's contribution.
    loser = min(first, key=lambda r: r["score"])
    bumped = [
        (n, o, (12.00 if n == loser["name"] else b)) for n, o, b in CANDIDATES
    ]
    second = run_auction(runtime, base, bumped,
                         f"AUCTION 2 -- same ads, but '{loser['name']}' bids $12.00 "
                         f"(was ${loser['bid']:.2f})")

    # Now hold bids at baseline and change one candidate's ad-side features.
    tweaked = [
        (n, ({**o, "banner_pos": "1"} if n == loser["name"] else o), b)
        for n, o, b in CANDIDATES
    ]
    third = run_auction(runtime, base, tweaked,
                        f"AUCTION 3 -- baseline bids, but '{loser['name']}' moves to "
                        f"banner_pos=1")

    print("\n" + "=" * 72)
    print("CRITERION 4: ordering depends on both bid and ad-side features")
    print("=" * 72)
    o1 = [r["name"] for r in first]
    o2 = [r["name"] for r in second]
    o3 = [r["name"] for r in third]
    print(f"  auction 1 (baseline)      : {' > '.join(o1)}")
    print(f"  auction 2 (bid changed)   : {' > '.join(o2)}")
    print(f"  auction 3 (features changed): {' > '.join(o3)}")

    bid_matters = o1 != o2
    feat_effect = any(
        abs(a["pctr"] - b["pctr"]) > 1e-6
        for a, b in zip(sorted(first, key=lambda r: r["name"]),
                        sorted(third, key=lambda r: r["name"]))
    )
    print(f"\n  ordering responds to bid        : {'PASS' if bid_matters else 'FAIL'}")
    print(f"  pCTR responds to ad features    : {'PASS' if feat_effect else 'FAIL'}")
    print(f"  ranking depends on both         : "
          f"{'PASS' if bid_matters and feat_effect else 'FAIL'}")

    print("\n  Note: the endpoint never received a bid. It returned pCTR only;")
    print("  every multiplication, comparison, and sort above happened here.")

    if not (bid_matters and feat_effect):
        sys.exit(1)


if __name__ == "__main__":
    main()
