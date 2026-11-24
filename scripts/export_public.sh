#!/bin/bash
# Build the public repository as ONE clean commit from the current branch's tree.
# The private development repo (full history, internal notes) is left untouched.
#
# Usage: scripts/export_public.sh OUT_DIR [REMOTE_URL]
#   OUT_DIR     new, empty directory for the export (created)
#   REMOTE_URL  optional; if given, adds it as `origin` (pushing is left to you)
#
# Refuses to export if any private file or marker is present in the tree.
# Personal markers are read from an UNTRACKED file (one extended regex, default
# ./.export_markers, override with EXPORT_MARKERS_FILE). It is gitignored so the
# terms it scrubs are never published. The script fails if the file is missing.
set -euo pipefail
OUT=${1:?usage: export_public.sh OUT_DIR [REMOTE_URL]}
REMOTE=${2:-}
SRC=$(git rev-parse --show-toplevel)
[ -e "$OUT" ] && { echo "refusing: $OUT already exists" >&2; exit 1; }

# 1. Paths that must never be published.
BLOCK='^(CLAUDE\.md|\.claude/|docs/open_questions\.md)|\.(safetensors|bin|pt)$|brainlm_cache/'
bad=$(git -C "$SRC" ls-files | grep -E "$BLOCK" || true)
[ -n "$bad" ] && { echo "refusing: private or licensed files tracked:" >&2; echo "$bad" >&2; exit 1; }

# 2. Personal markers that must not appear in any tracked file (list kept out of git).
MFILE=${EXPORT_MARKERS_FILE:-$SRC/.export_markers}
[ -s "$MFILE" ] || { echo "refusing: marker file $MFILE missing or empty (untracked; see header)" >&2; exit 1; }
git -C "$SRC" ls-files --error-unmatch "${MFILE#$SRC/}" >/dev/null 2>&1 && { echo "refusing: $MFILE is tracked by git" >&2; exit 1; }
MARK=$(tr -d '\n' < "$MFILE")
hits=$(git -C "$SRC" grep -I -i -l -E "$MARK" -- . || true)
[ -n "$hits" ] && { echo "refusing: personal markers found in:" >&2; echo "$hits" >&2; exit 1; }

# 3. Snapshot the committed tree (uncommitted changes are NOT exported).
mkdir -p "$OUT"
git -C "$SRC" archive HEAD | tar -x -C "$OUT"
cd "$OUT"
git init -q -b main
git add -A
git -c user.name="gammon-bio" -c user.email="226215021+gammon-bio@users.noreply.github.com" \
    commit -q -m "fMRI-model-weights: synthetic testbed and pre-registered agent repair studies

Public snapshot of the development repository (history not published)."
[ -n "$REMOTE" ] && git remote add origin "$REMOTE"
echo "exported $(git ls-files | wc -l) files to $OUT as one commit $(git rev-parse --short HEAD)"
[ -n "$REMOTE" ] && echo "next: git -C $OUT push -u origin main"
exit 0
