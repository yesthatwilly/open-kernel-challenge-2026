# Decode kernel blocks

A 3D view of `muse_decode_layer` across its 49 master landings, 2026-10-24 to 2026-11-04 (UTC). Open
[index.html](index.html) in a browser. It is one self-contained file with no network use.

## The two views

- **Where the time goes.** One block per kernel. Floor area is its GPU time per decode step, on one fixed
  scale, so the whole floor shrinks as the layer gets faster. Colour is the layer stage. Height is the
  speedup since the first landing of the block's group: attention, experts, or everything else. So each group's
  volume (area × height) always equals its first-landing time: speeding up squeezes a group (smaller floor, taller)
  instead of shrinking it, which keeps the biggest gains visible and makes speed a shape you can read.
- **Code against work.** One block per kernel. Width is lines of code, optionally including the helpers it
  calls. Height is GPU time per step. Depth is instructions executed per step. Depth is fixed until the GPU
  step below has run. Volume means nothing.

Below the picture, a results section gives the rank, the board and judged speedups, and a per-case table
(judge medians, speed-of-light, group speedups and time shares) whose rows open that case in the picture. `build.py`
reads it from the first and final landings' judge result files; the board figures are the closing board in the
top-level README.

The timeline plays the landings. Drag to turn, scroll to zoom, and hover a block for its numbers. The URL
hash keeps the view, shape and landing, e.g. `index.html#view=B&shape=0&pos=48`.

## Rebuild

```
python3 collect_times.py   # judge profiles -> data/times.json (stage rules in stage_map.tsv)
python3 count_loc.py       # git history of submission.py -> data/loc.json (about 4 minutes)
python3 build.py           # embeds data/*.json into page.template.html -> index.html
```

## Instruction counts (needs one GPU session)

`gpu/profile_one.py` runs one submission on the 12 judge cases with one profiled call per case. It runs
under Nsight Compute with `smsp__inst_executed.sum`. `gpu/lane-inst.sh` runs it for six milestone
landings: 164c1d27, 2196b906, d07539ae, 36584c5c, c3b07763 and c3ec85f7. It first checks that
performance counters are allowed and stops if they aren't. The placeholders in the lane are filled at
staging time. Results become `data/inst.json` as `{commit: {case index: {kernel base name: instructions}}}`,
and `build.py` picks them up.

## Known limits

- Five landings from 10-29 to 10-31 have no judge profile on record. The page holds the previous frame
  there and says so.
- Eight landings were matched to their scored job by speedup, not by a recorded job id. The page flags them.
- Shared cuBLAS GEMMs before 10-31, and a few twice-called kernels after it, can't be split by stage.
  They are grey ("unattributed"): 13% of time in our own era and about 5% after.
- Line counts include every `#if` branch, and the call graph matches identifiers, not resolved calls.
  See the header of `count_loc.py`.
