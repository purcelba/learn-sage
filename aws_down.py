"""One-command teardown: stop everything that bills, keep everything that doesn't.

    python aws_down.py              # report only (default)
    python aws_down.py --confirm    # stop the billing
    python aws_down.py --purge-all  # IRREVERSIBLE: also delete data and artifacts

## The distinction this script exists to make

Not everything left behind costs money, and treating it all alike is how you
either keep paying or destroy your own work:

  BILLS CONTINUOUSLY   endpoints, notebook instances, running jobs
  FREE (metadata)      models, endpoint configs, model packages, finished jobs
  ~FREE (~$0.0002/mo)  the S3 objects for this project

So `--confirm` deletes the first group and the endpoint configs, and leaves
everything else alone. **Teardown means stopping the billing, not deleting the
project.**

That is deliberate, not lazy. Phase 8 (Terraform) is built on importing the
resources that survive this script -- and the observation that those are exactly
the resources belonging in IaC is that phase's whole organizing idea. Nuking the
bucket would delete Phase 8's subject matter along with Phase 1's data.

## Three hazards learned the hard way

1. **`code/` is coupled to Models.** s3://<bucket>/code/ is not SDK scratch: it
   is where a deployed Model's source permanently lives
   (SAGEMAKER_SUBMIT_DIRECTORY). Deleting it between Phases 4 and 5 left a Model
   pointing at a tarball that no longer existed, and NOTHING reported the
   dangling reference -- CreateModel validates ModelDataUrl but evidently not
   the submit directory. So S3 purging is opt-in, and when used it deletes the
   dependent Models too rather than leaving them silently broken.

2. **Deletion is asynchronous.** An endpoint in `Deleting` is still billing.
   Reporting success before it is actually gone is a false all-clear, so this
   script polls until the count reaches zero.

3. **Resources hide in other regions.** The Phase 2 quota hunt probed us-west-2
   and us-east-2. A single-region teardown would miss anything stranded there.

It also checks for SageMaker Studio domains, which provision a persistent EFS
volume -- the classic thing a teardown written before it existed will not know
to look for.
"""

from __future__ import annotations

import argparse
import time

import boto3
from botocore.exceptions import ClientError

# Every region this project has ever touched, not just the one it lives in.
REGIONS = ["us-east-1", "us-west-2", "us-east-2"]
HOME_REGION = "us-east-1"
BUCKET_TEMPLATE = "learn-sage-{account_id}"


def sm(region: str):
    return boto3.client("sagemaker", region_name=region)


def survey(regions: list[str]) -> dict:
    """Collect state across regions. Read-only."""
    state = {}
    for region in regions:
        c = sm(region)
        state[region] = {
            "endpoints": [e["EndpointName"] for e in c.list_endpoints()["Endpoints"]],
            "configs": [x["EndpointConfigName"]
                        for x in c.list_endpoint_configs()["EndpointConfigs"]],
            "notebooks": [n["NotebookInstanceName"]
                          for n in c.list_notebook_instances()["NotebookInstances"]],
            "training": [j["TrainingJobName"] for j in c.list_training_jobs(
                StatusEquals="InProgress")["TrainingJobSummaries"]],
            "transform": [j["TransformJobName"] for j in c.list_transform_jobs(
                StatusEquals="InProgress")["TransformJobSummaries"]],
            "processing": [j["ProcessingJobName"] for j in c.list_processing_jobs(
                StatusEquals="InProgress")["ProcessingJobSummaries"]],
            "models": [m["ModelName"] for m in c.list_models()["Models"]],
        }
        try:
            state[region]["studio"] = [d["DomainId"] for d in c.list_domains()["Domains"]]
        except ClientError:
            state[region]["studio"] = []
    return state


def report(state: dict) -> int:
    """Print state; return how many billing resources are live."""
    billing = 0
    for region, s in state.items():
        live = (s["endpoints"] + s["notebooks"] + s["training"]
                + s["transform"] + s["processing"] + s["studio"])
        billing += len(live)
        free = s["configs"] + s["models"]
        if not live and not free:
            print(f"  {region:<12} clean")
            continue
        print(f"  {region}")
        for label, items, cost in [
            ("endpoints", s["endpoints"], "BILLING"),
            ("notebook instances", s["notebooks"], "BILLING"),
            ("training jobs (running)", s["training"], "BILLING"),
            ("transform jobs (running)", s["transform"], "BILLING"),
            ("processing jobs (running)", s["processing"], "BILLING"),
            ("studio domains", s["studio"], "BILLING (EFS)"),
            ("endpoint configs", s["configs"], "free"),
            ("models", s["models"], "free"),
        ]:
            if items:
                print(f"    {label:<26} {cost:<14} {items}")
    return billing


def stop_billing(state: dict) -> None:
    for region, s in state.items():
        c = sm(region)
        for name in s["endpoints"]:
            c.delete_endpoint(EndpointName=name)
            print(f"  [{region}] deleted endpoint {name} -- billing stopped")
        for name in s["configs"]:
            c.delete_endpoint_config(EndpointConfigName=name)
            print(f"  [{region}] deleted endpoint config {name}")
        for name in s["notebooks"]:
            # Must stop before delete; a running instance cannot be deleted.
            c.stop_notebook_instance(NotebookInstanceName=name)
            print(f"  [{region}] stopping notebook {name} (delete once Stopped)")
        for name in s["training"]:
            c.stop_training_job(TrainingJobName=name)
            print(f"  [{region}] stopping training job {name}")
        for name in s["transform"]:
            c.stop_transform_job(TransformJobName=name)
            print(f"  [{region}] stopping transform job {name}")
        for name in s["processing"]:
            c.stop_processing_job(ProcessingJobName=name)
            print(f"  [{region}] stopping processing job {name}")
        if s["studio"]:
            # Deleting a domain requires deleting its user profiles, apps and
            # spaces first -- too destructive to do implicitly.
            print(f"  [{region}] STUDIO DOMAINS PRESENT: {s['studio']}")
            print(f"           These hold a persistent EFS volume that bills. "
                  f"Delete via the console -- not automated here, since it "
                  f"requires removing user profiles and apps first.")


def wait_until_gone(regions: list[str], timeout: int = 900) -> bool:
    """Endpoints in `Deleting` still bill, so 'submitted' is not 'done'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = {r: sm(r).list_endpoints()["Endpoints"] for r in regions}
        total = sum(len(v) for v in remaining.values())
        if total == 0:
            return True
        names = [e["EndpointName"] for v in remaining.values() for e in v]
        print(f"  still deleting (these still bill): {names}")
        time.sleep(15)
    return False


def purge_s3(account_id: str, state: dict) -> None:
    """Delete data, artifacts, and the Models that depend on the code prefix."""
    bucket = BUCKET_TEMPLATE.format(account_id=account_id)
    s3 = boto3.resource("s3", region_name=HOME_REGION)

    # Models FIRST. Deleting code/ while Models reference it leaves them intact
    # but broken, pointing at a submit directory that no longer exists, with
    # nothing reporting it. Hazard 1 in the module docstring.
    for region, s in state.items():
        for name in s["models"]:
            sm(region).delete_model(ModelName=name)
            print(f"  [{region}] deleted model {name} (depends on s3://{bucket}/code/)")

    b = s3.Bucket(bucket)
    b.object_versions.delete()  # bucket is versioned; objects alone leave versions
    print(f"  emptied s3://{bucket} (including all versions)")
    print(f"  NOTE: the bucket itself is kept. Phase 8 (Terraform) imports it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="stop everything that bills")
    ap.add_argument("--purge-all", action="store_true",
                    help="IRREVERSIBLE: also delete S3 data, artifacts, and models")
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()

    account_id = boto3.client("sts", region_name=HOME_REGION) \
        .get_caller_identity()["Account"]

    print("=== current state ===")
    state = survey(args.regions)
    billing = report(state)
    print(f"\nbilling resources live: {billing}")

    if not (args.confirm or args.purge_all):
        print("\nReport only. Re-run with --confirm to stop billing.")
        return

    # Run unconditionally, NOT `if billing:`. Endpoint configs are free, so
    # gating cleanup on the billing count skipped them entirely whenever nothing
    # happened to be billing -- and then printed "CLEAN" with stale configs
    # listed directly above it. Caught by creating a throwaway config and
    # watching this script fail to remove it.
    print("\n=== tearing down ===")
    stop_billing(state)

    if any(s["endpoints"] for s in state.values()):
        print("\n=== waiting for deletion to complete ===")
        if not wait_until_gone(args.regions):
            print("  TIMED OUT -- endpoints still present. Re-run.")
            return

    if args.purge_all:
        print("\n=== PURGE: deleting data, artifacts, and dependent models ===")
        purge_s3(account_id, survey(args.regions))

    print("\n=== verification ===")
    final = survey(args.regions)
    remaining = report(final)
    stale = sum(len(s["configs"]) for s in final.values())

    print(f"\nbilling resources live: {remaining}")
    if remaining:
        print("SOMETHING IS STILL BILLING. Re-run.")
    elif stale:
        # Distinguished on purpose: the earlier version printed "CLEAN" with
        # stale configs listed directly above it. True about billing, misleading
        # about state.
        print(f"Nothing is billing, but {stale} endpoint config(s) remain. Re-run.")
    else:
        print("CLEAN -- nothing is billing, nothing stale.")

    if not args.purge_all:
        bucket = BUCKET_TEMPLATE.format(account_id=account_id)
        print(f"\nDeliberately kept (Phase 8 imports these; ~$0.0002/month):")
        print(f"  s3://{bucket}          data, model artifact, predictions")
        print(f"  IAM role + policies    AmazonSageMaker-learn-sage-ExecutionRole")
        print(f"  model registry         learn-sage-pctr")
        print(f"  Models                 free metadata; keep code/ while these exist")


if __name__ == "__main__":
    main()
