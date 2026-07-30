"""Delete the endpoint and everything that hangs off it.

Written BEFORE deploy_endpoint.py was ever run, deliberately: the way to stop
billing should exist before the billing starts.

Deletes three distinct objects, because deleting only the first is the classic
mistake:

  1. Endpoint        -- the running instance. THIS is what bills.
  2. EndpointConfig  -- the instance-type/model recipe. Free, but it lingers
                        invisibly and blocks reusing the same name cleanly.
  3. Model           -- metadata. Free. Optional; kept by default since Phase 3
                        created one deliberately and Phase 5 may reuse it.

Only #1 costs money. #2 and #3 are hygiene -- but "no endpoints listed" while
stale configs pile up is exactly the false sense of tidiness Phase 6 is meant to
eliminate.

Usage:
    python teardown_endpoint.py                 # show what exists
    python teardown_endpoint.py --confirm       # delete endpoint + config
    python teardown_endpoint.py --confirm --all # also delete the Model objects
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
ENDPOINT_NAME = "learn-sage-pctr"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="also delete Model objects (metadata, no cost)")
    args = ap.parse_args()

    sm = boto3.Session(region_name=REGION).client("sagemaker")

    endpoints = sm.list_endpoints()["Endpoints"]
    configs = sm.list_endpoint_configs()["EndpointConfigs"]
    models = sm.list_models()["Models"]

    print(f"endpoints        : {[e['EndpointName'] for e in endpoints] or 'none'}  <- the billing ones")
    print(f"endpoint configs : {[c['EndpointConfigName'] for c in configs] or 'none'}")
    print(f"models           : {[m['ModelName'] for m in models] or 'none'}")

    if not args.confirm:
        print("\nNothing deleted. Re-run with --confirm.")
        return

    for ep in endpoints:
        sm.delete_endpoint(EndpointName=ep["EndpointName"])
        print(f"deleted endpoint {ep['EndpointName']} -- billing stopped")

    for cfg in configs:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"deleted endpoint config {cfg['EndpointConfigName']}")

    if args.all:
        for m in models:
            try:
                sm.delete_model(ModelName=m["ModelName"])
                print(f"deleted model {m['ModelName']}")
            except ClientError as err:
                print(f"could not delete model {m['ModelName']}: {err}")

    print("\n--- verification ---")
    remaining = sm.list_endpoints()["Endpoints"]
    print(f"endpoints remaining: {[e['EndpointName'] for e in remaining] or 'none'}")
    if remaining:
        # Deletion is asynchronous; an endpoint in state Deleting still bills
        # until it is actually gone. Reporting "done" here would be wrong.
        print("STILL DELETING -- re-run to confirm they are gone. These still bill.")
    else:
        print("Nothing is billing.")


if __name__ == "__main__":
    main()
