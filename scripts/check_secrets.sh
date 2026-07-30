#!/usr/bin/env bash
#
# Scan repo files for credentials, tokens, and the AWS account ID before they
# reach a public remote.
#
#   scripts/check_secrets.sh            # staged files only (what a commit would add)
#   scripts/check_secrets.sh --all      # every tracked file (use before pushing)
#
# Installed as a pre-commit hook by scripts/install-hooks.sh, so this runs
# automatically rather than depending on anyone remembering it.
#
# Design note: this script must never print the account ID it is protecting --
# it reads the value at runtime from STS and only ever reports *that* a match
# occurred, never the matched text. A leak-checker that echoes the secret into
# CI logs has just moved the leak.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

MODE="${1:-staged}"
FAIL=0

if [[ "$MODE" == "--all" ]]; then
  FILES=$(git ls-files)
  echo "Scanning all tracked files..."
else
  FILES=$(git diff --cached --name-only --diff-filter=ACM)
  echo "Scanning staged files..."
fi

# Nothing staged is a pass, not an error (e.g. a commit with only deletions).
if [[ -z "$FILES" ]]; then
  echo "  (no files to scan)"
fi

report() {
  echo ""
  echo "  BLOCKED: $1"
  echo "     $2"
  FAIL=1
}

# --- 1. Credential patterns -------------------------------------------------
# Each entry is "label|regex". Kept deliberately tight: a checker that cries
# wolf gets disabled, and a disabled checker protects nothing.
PATTERNS=(
  "AWS access key ID|AKIA[0-9A-Z]{16}"
  "AWS temporary key ID|ASIA[0-9A-Z]{16}"
  "AWS secret access key assignment|aws_secret_access_key[[:space:]]*=[[:space:]]*[A-Za-z0-9/+=]{40}"
  "Kaggle API token|KGAT_[A-Za-z0-9]{16,}"
  "GitHub token|(ghp|gho|ghu|ghs)_[A-Za-z0-9]{36}"
  "GitHub fine-grained PAT|github_pat_[A-Za-z0-9_]{22,}"
  "Private key block|-----BEGIN[A-Z ]*PRIVATE KEY-----"
  "Slack token|xox[baprs]-[A-Za-z0-9-]{10,}"
)

for entry in "${PATTERNS[@]}"; do
  label="${entry%%|*}"
  regex="${entry#*|}"
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    if grep -EqI "$regex" "$f" 2>/dev/null; then
      report "$label found in $f" "Remove it, then rotate the credential -- deleting the file does not un-leak a pushed secret."
    fi
  done <<< "$FILES"
done

# --- 2. AWS account ID ------------------------------------------------------
# Resolved at runtime so the ID never appears in this file. Skipped (with a
# warning, not a failure) when AWS credentials aren't available -- e.g. in CI.
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [[ -n "$ACCOUNT_ID" && "$ACCOUNT_ID" =~ ^[0-9]{12}$ ]]; then
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    if grep -qIF "$ACCOUNT_ID" "$f" 2>/dev/null; then
      report "AWS account ID found in $f" "Substitute the ACCOUNT_ID placeholder, or derive it at runtime via sts:GetCallerIdentity."
    fi
  done <<< "$FILES"
else
  echo "  WARN: could not resolve AWS account ID (no credentials?) -- skipping that check."
fi

# --- 3. Ignore rules still intact -------------------------------------------
# The pattern checks above only see files git knows about. These confirm the
# directories holding raw secrets and data are still excluded in the first
# place -- a .gitignore edit could silently undo that.
for d in secrets/ data/; do
  if ! git check-ignore -q "$d" 2>/dev/null; then
    report "$d is NOT git-ignored" "Restore the rule in .gitignore before committing."
  fi
done

# --- 4. Nothing from those directories is already tracked -------------------
TRACKED_BAD=$(git ls-files | grep -E "^(secrets/|data/)" || true)
if [[ -n "$TRACKED_BAD" ]]; then
  report "files tracked from an ignored directory:" "$TRACKED_BAD"
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "PASS: no credentials, tokens, or account IDs found."
  exit 0
fi
echo "FAILED. Commit blocked. Bypass with --no-verify only if you are certain."
exit 1
