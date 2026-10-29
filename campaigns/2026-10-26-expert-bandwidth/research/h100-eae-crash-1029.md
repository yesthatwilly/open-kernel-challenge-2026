# H100 triage: master moe [ea-enc] sticky illegal address in prepare, 2026-10-29 (gpunode06 GPU 4, 23:20-23:53Z)

Tooling: takemyjob `hopper/eaecrash-1029` (worktree `~/git/.takemyjob-wt/eaecrash`, branched from glue-1029 f3d0fea,
uncommitted configs `eae-*`, snaps in `problems/{moe,decode-jl}/snap/`), bundle `HP_BUNDLE=njeae`. Rules: USER,
no root, no Slurm allocation, guarded `hopper-probe run gpunode06 --gpus 4`, node-local only, warm nodejudge env read-only.
Nothing is left running.

## Verdict

**The bug is not at T=1.** `_eae_base` already returns `(None, "route v1 not FAST K1+K2")` for v1, and forcing v1 or
v6pe2 at T=1 passes. The crash is a **use-after-free in the T=256 cold-half iso graphs**:

- `muse_moe_w4a16/submission.py:18462-18466` (master 09b11a9d) has `_eae_class` capture `iso[("c", a)]` over
  `h3 / y3 / l3`. Those buffers are allocated outside the capture, and only the lambda's default args reference them.
- `_rt_graph` keeps only the graph, so each set is freed when the arm loop rebinds `h3 / y3 / l3`. The graph still
  writes to those addresses in `_rt_time`.
- On gpunode06 the first warm replay of `("c", None)` faults (line 4301, the sync after the warm replays). Then the
  context is poisoned and prepare fails.
- The fault depends on memory layout. It shows only with the sglang baseline party in the same process, which is the
  layout of gl-plain and of the judge. On the judge the UAF has written into freed cached blocks (harmless so far),
  and any run could fault the same way. **It hits every moe run whose T=256 route is the hybrid cold half that
  `_eae_base` accepts**, which is both the judge's v4cxs and the node's v4xs. It does not depend on the T=1 route.

## Evidence (each a guarded GPU 4 run; prepare-only shapes decode 1 / verify 1, reps 2)

| run | bytes | baseline in process | result |
|---|---|---|---|
| 162013 | dbg (sync markers), nat + forced v6pe2 | no | pass, pass |
| 162628 | master 09b11a9d | no | pass |
| 162823 | master 09b11a9d | **yes** | **FAIL** `[ea-enc] raised AcceleratorError ... illegal memory access` |
| 162948 | dbg (sync markers) | yes | **FAIL**; traceback `_eae_class -> _rt_time` warm replay, after T=256 p10 "iso ok" |
| 1632xx | dbg2 (sync after each graph replay) | yes | **FAIL** after `replay None/3/4 ok`, i.e. on the first `('c', None)` iso replay |
| 1634xx | dbg2 + keep h3/y3/l3 alive | yes | **pass**; all 3 T=256 draws `replay ('c', *) ok` |

Together with the glue lane's 2/2, the unfixed master fails 5 of 5 runs with the baseline in process.

## Fix: `hyp/muse_moe_w4a16/eae-t1-guard` b743b4e4 (worktree `.worktrees/muse_moe_w4a16/eae-t1-guard`)

3 lines: `keep = []`, `keep.append((h3, y3, l3))` before each iso capture, `del ..., keep` after timing. The file is
948,330 B, sha256 986a2bad. The timed path, routes and picks are unchanged: `crat` (the cold iso ratio) is print-only,
and the T=1 guard was already correct. Validation with the baseline in process:

| run | T=1 route | result | [ea-enc] PATH CHOSEN |
|---|---|---|---|
| 163643 | natural (v1) | pass, x1.467 | all master (node is HBM-walled; E/m > 0.98) |
| 163800 | forced v1 | pass, x1.447 | T=1 master (route v1 not FAST K1+K2), others master |
| 163910 | forced v6pe2 | pass, x1.498 | T=1 raced (8 arms, all exact y), master on ratio |
| 164020 | natural (v1) | pass, x1.438 | all master |

Every raced arm in every run was exact (`y`). The prepares took 31-43 s on warm caches.

## Decode (hyp/muse_decode_layer/engine-a-encodings-wh5ds-decode-stack, e1a44258)

**No hazard.**
- Its `_eae_class` captures `_rt_graph(_eae_fn(...))`, where every buffer is allocated inside the capture or is a live
  local (`d`). It has no external-buffer iso graphs, and its cold path runs `_hc_experts` inside the capture.
- Its T=1 guard is the same as moe's.
- Runs 164219 (natural, v6pe2 on the node), 164924 (forced v6pe2) and 165108 (forced v1): all pass with the baseline
  in process, with PATH lines printed and all arms exact.
- No guard is needed.

## Follow-ups
- Land eae-t1-guard on master moe before the final window: a check run, or ride the next moe submit (no perf effect).
- Other external-buffer graphs in moe (`_wh_time` `out`, `_wh_replay` `outs`) stay referenced through `_rt_time`.
  They are fine.
