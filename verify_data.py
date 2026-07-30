"""Phase 1 acceptance criteria, run as assertions.

Criterion 1: the bucket lists train.csv and test.csv.
Criterion 2: a read-back confirms row counts match the split and headers are
             intact.

The read-back deliberately pulls the objects *from S3* rather than reading the
local files. Re-reading data/train.csv would only prove pandas can read what
pandas just wrote -- it would say nothing about whether the upload landed
correctly, which is the thing the criterion is actually about.
"""

import io

import boto3
import pandas as pd

REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"

# The `id` column is a row identifier, not a feature, and Phase 1 drops it.
# Its presence downstream would silently give the model a near-unique key to
# memorize, so this is worth asserting rather than assuming.
FORBIDDEN_COLS = {"id"}
EXPECTED = {
    "data/train/train.csv": 40_000,
    "data/test/test.csv": 10_000,
}


def main() -> None:
    session = boto3.Session(region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    s3 = session.client("s3")

    print(f"=== Criterion 1: objects present in s3://{bucket} ===")
    listing = s3.list_objects_v2(Bucket=bucket, Prefix="data/")
    keys = {o["Key"]: o["Size"] for o in listing.get("Contents", [])}
    for key in EXPECTED:
        assert key in keys, f"MISSING: s3://{bucket}/{key}"
        print(f"  OK  s3://{bucket}/{key}  ({keys[key] / 1e6:.2f} MB)")

    print(f"\n=== Criterion 2: read-back from S3 ===")
    headers = {}
    for key, expected_rows in EXPECTED.items():
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(io.BytesIO(body), dtype=str)

        assert len(df) == expected_rows, (
            f"{key}: expected {expected_rows:,} rows, got {len(df):,}"
        )
        leaked = FORBIDDEN_COLS & set(df.columns)
        assert not leaked, f"{key}: dropped column(s) present: {leaked}"
        assert "click" in df.columns, f"{key}: label column `click` missing"

        headers[key] = list(df.columns)
        click_rate = df["click"].astype(int).mean()
        print(f"  OK  {key}: rows={len(df):,}  cols={df.shape[1]}  "
              f"click_rate={click_rate:.4f}")

    # Training and test schemas must be identical -- a column-order difference
    # here would surface in Phase 2 as a confusing feature-mismatch error.
    (train_key, train_cols), (test_key, test_cols) = headers.items()
    assert train_cols == test_cols, (
        f"header mismatch:\n  {train_key}: {train_cols}\n  {test_key}: {test_cols}"
    )
    print(f"  OK  headers identical across train/test ({len(train_cols)} columns)")
    print(f"\n  Columns: {train_cols}")

    print("\nPhase 1 acceptance criteria: PASS")


if __name__ == "__main__":
    main()
