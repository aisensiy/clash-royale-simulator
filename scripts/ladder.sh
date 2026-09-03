#!/bin/bash
# Acceptance ladder for a new checkpoint against the current reference.
#
#   ./ladder.sh /path/to/new_model.zip [games]
#
# Does what used to take a morning of hand-holding:
#   1. builds a throwaway ladder dir: reference.zip + era8stack context + the new model
#   2. runs elo.py (games per pairing, default 100) and behaviour.py
#   3. computes the head-to-head vs the reference with an exact binomial p-value
#   4. prints a promotion verdict and writes logs under checkpoints/logs/
#
# Promotion rule (issue #10 revision of the original +50 bar): promote when the
# same-ladder gap is >= +40 AND the head-to-head p-value < 0.05. The bare Elo
# number alone sent #7 to a coin flip; significance is what the +50 was trying
# to approximate.
set -euo pipefail

NEW_MODEL=$1
GAMES=${2:-100}
NAME=$(basename "$NEW_MODEL" .zip)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLasher=$ROOT/src/clasher_new
CKPT=$ROOT/checkpoints
PY=$ROOT/.venv/bin/python
STAMP=$(date +%Y%m%d-%H%M)
LADDER=$CKPT/ladder_auto_$STAMP
mkdir -p "$LADDER" "$CKPT/logs"

[ -f "$CKPT/reference.zip" ] || { echo "no reference.zip in $CKPT"; exit 1; }
cp "$CKPT/reference.zip" "$LADDER/0_reference.zip"
[ -f "$CKPT/ladder_all/era8_stack_stacked_final.zip" ] && \
  cp "$CKPT/ladder_all/era8_stack_stacked_final.zip" "$LADDER/1_ctx_era8stack.zip"
cp "$NEW_MODEL" "$LADDER/2_candidate_$NAME.zip"

echo "== ladder: $GAMES games/pairing =="
cd "$CLasher"
"$PY" elo.py "$LADDER" --games "$GAMES" --workers 10 2>&1 | tee "$CKPT/logs/ladder_${NAME}_$STAMP.log"

echo "== behaviour vs rusher =="
"$PY" behaviour.py "$LADDER/2_candidate_$NAME.zip" "$LADDER/0_reference.zip" \
  --opponent rusher --games 40 --workers 10 2>&1 | tee "$CKPT/logs/behaviour_${NAME}_$STAMP.log"

echo "== verdict =="
"$PY" - "$CKPT/logs/ladder_${NAME}_$STAMP.log" "$GAMES" <<'PYEOF'
import math, re, sys

log = open(sys.argv[1]).read()
games = int(sys.argv[2])

rows = re.findall(r"^(\S+\.zip)\s+([\d.]+)\s*$", log, re.M)
ratings = {n: float(e) for n, e in rows}
ref = next((n for n in ratings if n.startswith("0_reference")), None)
cand = next((n for n in ratings if n.startswith("2_candidate")), None)
if not ref or not cand:
    print("!! could not find both players in the ladder table"); sys.exit(1)

gap = ratings[cand] - ratings[ref]
m = re.search(re.escape(cand) + r" vs " + re.escape(ref) + r": ([\d.]+)/(\d+)", log)
if not m:
    print("!! no direct head-to-head pairing was played"); sys.exit(1)
wins = float(m.group(1))

# exact one-sided binomial tail: P(X >= wins | n, p=0.5)
n = int(m.group(2)); w = int(round(wins))
pval = sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n

sig = "YES" if pval < 0.05 else "no"
print(f"candidate : {cand}  ({ratings[cand]:.0f})")
print(f"reference : {ref}  ({ratings[ref]:.0f})")
print(f"gap       : {gap:+.0f}   H2H {w}/{n}   p={pval:.4f}  (significant: {sig})")
if gap >= 40 and pval < 0.05:
    print("VERDICT   : PROMOTE — candidate clears the rule (>=+40 and significant)")
elif pval < 0.05:
    print("VERDICT   : significant but under the +40 bar — maintainer call, default NO")
else:
    print("VERDICT   : NO PROMOTION — not distinguishable from the reference")
PYEOF
echo "== logs saved under checkpoints/logs/ =="
