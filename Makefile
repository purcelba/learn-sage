# Convenience targets. The scripts are the real interface; these just spell
# the common invocations so teardown is never a remembered checklist.

PY := .venv/bin/python
export SAGEMAKER_SUPPRESS_V2_WARNING = 1

.PHONY: help aws-status aws-down aws-purge check-secrets

help:
	@echo "aws-status     what exists and what of it bills (read-only)"
	@echo "aws-down       stop everything that bills; keep data and artifacts"
	@echo "aws-purge      IRREVERSIBLE: aws-down plus delete data, artifacts, models"
	@echo "check-secrets  scan all tracked files for credentials and account IDs"

# Read-only. Safe to run any time, and the right thing to run at the end of a
# session before closing the laptop.
aws-status:
	@$(PY) aws_down.py

# The one command Phase 6 asks for. Deletes endpoints, endpoint configs, and
# stops running jobs, across every region this project has touched. Keeps the
# bucket, the IAM role, the model artifact, and the registry -- teardown means
# stopping the billing, not deleting the project. Phase 8 imports what survives.
aws-down:
	@$(PY) aws_down.py --confirm

# For when you are genuinely finished. Deletes the data, the model artifact, the
# predictions, and the Models that depend on s3://<bucket>/code/. Deletes the
# Models FIRST, because removing code/ while a Model references it leaves the
# Model intact but silently broken.
aws-purge:
	@echo "This deletes your data and model artifacts. Ctrl-C within 5s to abort."
	@sleep 5
	@$(PY) aws_down.py --purge-all

check-secrets:
	@./scripts/check_secrets.sh --all
