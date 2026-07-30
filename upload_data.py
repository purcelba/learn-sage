"""Create the project's one S3 bucket and upload the Phase 1 CSVs.

This is the first thing in the project that touches billable AWS. The cost is
negligible (~7 MB of Standard storage plus a handful of PUT requests -- inside
free tier, fractions of a cent otherwise), but it is nonzero, so it lives behind
an explicit --confirm flag rather than running as a side effect.

Per CLAUDE.md: ONE bucket for the whole project, data in and artifacts out.
Phase 2's training job reads from data/train/ and writes to output/.

Usage:
    python upload_data.py --dry-run    # show what would happen, no AWS writes
    python upload_data.py --confirm
"""

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
# Bucket names are globally unique across all of AWS, so the account ID suffix
# avoids collision with anyone else's "learn-sage".
BUCKET_TEMPLATE = "learn-sage-{account_id}"

UPLOADS = [
    ("data/train.csv", "data/train/train.csv"),
    ("data/test.csv", "data/test/test.csv"),
]


def resolve_bucket(session: boto3.Session) -> str:
    account_id = session.client("sts").get_caller_identity()["Account"]
    return BUCKET_TEMPLATE.format(account_id=account_id)


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket s3://{bucket} already exists -- reusing it.")
        return
    except ClientError as err:
        code = err.response["Error"]["Code"]
        if code not in ("404", "NoSuchBucket"):
            # 403 here means the name is taken by another AWS account, which is a
            # different problem than "doesn't exist yet" and needs a new name.
            raise

    # us-east-1 is special: passing CreateBucketConfiguration with
    # LocationConstraint=us-east-1 is an InvalidLocationConstraint error, unlike
    # every other region where it's required.
    s3.create_bucket(Bucket=bucket)
    print(f"Created bucket s3://{bucket} in {REGION}")

    # Not strictly needed for a sandbox, but versioning is cheap insurance
    # against overwriting train.csv with a bad regeneration mid-project.
    s3.put_bucket_versioning(
        Bucket=bucket, VersioningConfiguration={"Status": "Enabled"}
    )
    print("Enabled versioning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without creating or uploading anything")
    ap.add_argument("--confirm", action="store_true",
                    help="required to actually write to AWS")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    bucket = resolve_bucket(session)

    print(f"Region: {REGION}")
    print(f"Bucket: s3://{bucket}")
    for local, key in UPLOADS:
        print(f"  {local}  ->  s3://{bucket}/{key}")

    if args.dry_run or not args.confirm:
        print("\nNo AWS writes performed. Re-run with --confirm to upload.")
        sys.exit(0)

    s3 = session.client("s3")
    ensure_bucket(s3, bucket)

    for local, key in UPLOADS:
        s3.upload_file(local, bucket, key)
        print(f"Uploaded {local} -> s3://{bucket}/{key}")

    print("\nDone. Verify with: python verify_data.py")


if __name__ == "__main__":
    main()
