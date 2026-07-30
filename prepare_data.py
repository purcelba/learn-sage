"""Carve a 50k-row sample out of the full Avazu train file and split it.

Phase 1 only. This script deliberately does NOT encode anything: it drops `id`,
samples, splits, and writes plain CSV with headers. Categorical encoding is
`train.py`'s job in Phase 2, so that training and serving share exactly one
implementation of "raw column -> feature" (see CLAUDE.md, feature parity).
In particular `hour` stays as its raw YYMMDDHH string here -- extracting
hour-of-day is a modeling decision, not a data-prep one.

Usage:
    python prepare_data.py                    # defaults below
    python prepare_data.py --n-rows 50000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# The one column that is a row identifier rather than a feature. `click` is the
# label and stays in the CSV; Phase 2 separates X from y.
DROP_COLS = ["id"]

# Avazu's train file is ordered by time across 10 days. Reading the first N rows
# would give us a couple of hours of day one, which would make the hour-of-day
# context feature nearly constant and useless in Phase 2. So we sample
# systematically across the whole file instead: keep every COARSE_STRIDE'th row
# on a streaming pass, then thin that pool down to exactly --n-rows.
#
# Two stages rather than one because the exact total row count (~40.4M) isn't
# known until the pass finishes. A single stride tuned to land on exactly 50k
# would come up short if the file were smaller than expected; over-collecting
# and then thinning is robust in both directions.
COARSE_STRIDE = 400
CHUNK_SIZE = 500_000


def collect_sample(raw_path: Path) -> pd.DataFrame:
    """Stream the gzipped train file, keeping every COARSE_STRIDE'th row."""
    kept: list[pd.DataFrame] = []
    total_seen = 0

    # dtype=str across the board: site_id, device_ip, device_model et al are hex
    # strings whose leading zeros are significant, and the anonymized C* columns
    # are categorical despite looking numeric. Letting pandas infer types would
    # silently coerce some of them to ints and lose information. `click` is cast
    # back to int below, where it's actually needed as a number.
    reader = pd.read_csv(
        raw_path,
        compression="gzip",
        dtype=str,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        # Offset the stride by our position in the overall file so the sampled
        # rows stay evenly spaced across chunk boundaries rather than clustering
        # at the start of each chunk.
        offset = (-total_seen) % COARSE_STRIDE
        kept.append(chunk.iloc[offset::COARSE_STRIDE])
        total_seen += len(chunk)
        print(f"  scanned {total_seen:,} rows, pool at {sum(len(k) for k in kept):,}")

    pool = pd.concat(kept, ignore_index=True)
    print(f"\nScanned {total_seen:,} rows total; sample pool = {len(pool):,}")
    return pool


def thin_to(pool: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Take exactly n_rows evenly spaced rows from the pool, preserving order."""
    if len(pool) < n_rows:
        raise SystemExit(
            f"Sample pool has only {len(pool):,} rows, need {n_rows:,}. "
            f"Lower COARSE_STRIDE (currently {COARSE_STRIDE}) and re-run."
        )
    idx = np.linspace(0, len(pool) - 1, num=n_rows).astype(int)
    return pool.iloc[idx].reset_index(drop=True)


def describe(name: str, df: pd.DataFrame) -> None:
    click_rate = df["click"].astype(int).mean()
    print(f"  {name:<6} rows={len(df):>7,}  cols={df.shape[1]:>3}  click_rate={click_rate:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/train.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-rows", type=int, default=50_000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"Missing {args.raw}. Download it first with:\n"
                         f"  kaggle competitions download -c avazu-ctr-prediction "
                         f"-f train.gz -p {args.raw.parent}/")

    print(f"Streaming {args.raw} (every {COARSE_STRIDE}th row)...")
    pool = collect_sample(args.raw)
    sample = thin_to(pool, args.n_rows)

    dropped = [c for c in DROP_COLS if c in sample.columns]
    sample = sample.drop(columns=dropped)
    print(f"Dropped {dropped} -> {sample.shape[1]} columns "
          f"(1 label + {sample.shape[1] - 1} features)")

    # Stratify on the label: at ~17% base CTR, an unstratified split can drift
    # the positive rate between train and test, which would make the Phase 2
    # test AUC harder to read.
    train_df, test_df = train_test_split(
        sample,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=sample["click"].astype(int),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nWrote:")
    describe("train", train_df)
    describe("test", test_df)
    print(f"\n  {train_path}  ({train_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {test_path}  ({test_path.stat().st_size / 1e6:.2f} MB)")
    print(f"\nColumns: {list(train_df.columns)}")


if __name__ == "__main__":
    main()
