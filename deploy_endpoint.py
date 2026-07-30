"""Phase 4: deploy the trained model behind a real-time HTTPS endpoint.

*** THIS IS THE ONE THING IN THIS PROJECT THAT BILLS CONTINUOUSLY. ***

Training jobs and Batch Transform jobs terminate themselves. An endpoint does
not: it holds a dedicated instance until explicitly deleted, whether it serves
one request an hour or none at all. ml.t2.medium is ~$0.065/hr = ~$1.56/day =
~$47/month, against a $15 budget. Delete it as soon as the phase's criteria
pass -- `python teardown_endpoint.py --confirm`.

## Why this is NOT how Lyft serves models

LyftLearn Serving runs on Kubernetes; only LyftLearn Compute (training, batch,
notebooks) maps to SageMaker. What this script demonstrates is precisely why
that choice gets made: one always-on instance, dedicated to one model, idle most
of the time, billing regardless. Fifty models means fifty instances. A shared
Kubernetes cluster amortizes capacity across all of them and can scale down
between requests. The endpoint below is the thing that tradeoff is measured
against, not the pattern to copy.

Usage:
    python deploy_endpoint.py --dry-run
    python deploy_endpoint.py --confirm
"""

from __future__ import annotations

import argparse
import sys

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"
ROLE_NAME = "AmazonSageMaker-learn-sage-ExecutionRole"
TRAINING_JOB = "learn-sage-pctr-2026-07-30-19-35-01-384"

ENDPOINT_NAME = "learn-sage-pctr"
# NOT ml.t3.medium, which PHASES.md specifies: t3 is not a supported
# real-time endpoint instance type (CreateEndpointConfig rejects it, and
# that is also why no "ml.t3.medium for endpoint usage" quota exists).
# ml.t2.medium is the nearest equivalent -- same burstable family, one
# generation older -- and cheaper than the ml.m5.large alternative.
INSTANCE_TYPE = "ml.t2.medium"
USD_PER_HOUR = 0.065

# Same image family and version as training and as the Phase 3 Model. The
# artifact was pickled by scikit-learn 1.4.2; an older runtime cannot load it.
FRAMEWORK_VERSION = "1.4-2"


def resolve_role_arn(session: boto3.Session) -> str:
    iam = session.client("iam")
    for page in iam.get_paginator("list_roles").paginate():
        for role in page["Roles"]:
            if role["RoleName"] == ROLE_NAME:
                return role["Arn"]
    raise SystemExit(f"Role {ROLE_NAME!r} not found -- see iam/sagemaker-execution-role.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="required -- this starts a continuously billing resource")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    role_arn = resolve_role_arn(session)
    artifact = f"s3://{bucket}/output/{TRAINING_JOB}/output/model.tar.gz"

    print(f"endpoint    : {ENDPOINT_NAME}")
    print(f"instance    : {args.instance_type}")
    print(f"artifact    : {artifact}")
    print(f"entry point : src/inference.py (ships src/features.py alongside it)")
    print(f"framework   : SKLearn {FRAMEWORK_VERSION}")
    print()
    print(f"*** BILLS CONTINUOUSLY: ~${USD_PER_HOUR}/hr, ~${USD_PER_HOUR * 24:.2f}/day ***")
    print(f"    Delete with: python teardown_endpoint.py --confirm")

    if args.dry_run or not args.confirm:
        print("\nNothing deployed. Re-run with --confirm.")
        sys.exit(0)

    sm_session = sagemaker.Session(boto_session=session)

    model = SKLearnModel(
        model_data=artifact,
        role=role_arn,
        entry_point="inference.py",
        # Ships features.py into the container, so serving imports the SAME
        # module training used rather than a copy that can drift.
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=sm_session,
        code_location=f"s3://{bucket}/code",
        name=f"{ENDPOINT_NAME}-model",
    )

    print("\nDeploying (typically 5-10 minutes)...")
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"\nendpoint '{ENDPOINT_NAME}' is InService and BILLING NOW.")
    print(f"Tear it down as soon as you're done: python teardown_endpoint.py --confirm")


if __name__ == "__main__":
    main()
