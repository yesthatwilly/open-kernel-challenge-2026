#!/usr/bin/env bash
# Instruction counts for the six milestone landings. Runs on ONE H100/H200 inside an allocation the
# user opened (srun --jobid=<id> --overlap). Writes only under $OUT.
#
#   Reads (never writes): $D/venv314 (python with torch+sglang), $D/h100/decode_layer (reference.py),
#                         $D/h100/dec-cases.json (the 12 judge cases)
#   Writes: $OUT/<commit>.py (the milestone file), $OUT/<commit>.csv, $OUT/<commit>.log, $OUT/ext/
set -u
D=/scratch/USER_PLACEHOLDER/okc-h100-20261103
OUT=/scratch/USER_PLACEHOLDER/okc-viz-20261105
NCU=/opt/apps/NCU_PLACEHOLDER/ncu
mkdir -p "$OUT/ext"
export TORCH_EXTENSIONS_DIR=$OUT/ext TORCH_CUDA_ARCH_LIST=9.0a
module load cuda-13.0.1-gcc-12.1.0 2>/dev/null || true
cd "$D/h100/decode_layer"
echo "start $(date -u +%T) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "$OUT/lane.log"

# Gate: one tiny profiled kernel. Stop if performance counters are not allowed for this user.
"$NCU" --metrics smsp__inst_executed.sum --csv "$D/venv314/bin/python" -c 'import torch; torch.ones(8,device="cuda").sum(); torch.cuda.synchronize()' \
  > "$OUT/gate.log" 2>&1
if grep -q 'ERR_NVGPUCTRPERM' "$OUT/gate.log" || ! grep -q 'inst_executed' "$OUT/gate.log"; then
  echo "GATE FAILED: performance counters unavailable, see gate.log" >> "$OUT/lane.log"; exit 3
fi
echo "gate ok $(date -u +%T)" >> "$OUT/lane.log"

for c in $(cat "$OUT/milestones.txt"); do
  f=$OUT/$c.py
  [ -s "$f" ] || { echo "missing $f" >> "$OUT/lane.log"; continue; }
  timeout 2400 "$NCU" --profile-from-start off --graph-profiling node --nvtx \
    --metrics smsp__inst_executed.sum,gpu__time_duration.sum --csv --page raw \
    --log-file "$OUT/$c.csv" \
    "$D/venv314/bin/python" "$OUT/profile_one.py" "$D/h100/decode_layer" "$f" "$D/h100/dec-cases.json" \
    > "$OUT/$c.log" 2>&1
  echo "$c rc=$? $(date -u +%T)" >> "$OUT/lane.log"
done
echo "done $(date -u +%T)" >> "$OUT/lane.log"
