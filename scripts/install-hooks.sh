#!/usr/bin/env bash
#
# Install the pre-commit secret scan.
#
# Git hooks live in .git/hooks, which is NOT version controlled -- so a fresh
# clone has no hooks until this is run. That's the one gap in the "enforced,
# not remembered" design, and the reason this script exists rather than the
# hook simply being committed.
#
#   scripts/install-hooks.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p .git/hooks
cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/check_secrets.sh"
HOOK

chmod +x .git/hooks/pre-commit
chmod +x scripts/check_secrets.sh

echo "Installed .git/hooks/pre-commit -> scripts/check_secrets.sh"
echo "Every commit now scans staged files. Run scripts/check_secrets.sh --all before pushing."
