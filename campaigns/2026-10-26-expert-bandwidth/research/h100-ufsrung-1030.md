# H100 lane: the UFS reopen rung on moe verify 32 (bs 256), 2026-10-30 (gpunode06 GPU 1, 02:50Z-03:07Z)

Pre-registered in h100-m256-1030.md §4, reopen condition 2: "an nsys + gpu-metrics pass on the board bytes, showing
judge-scaled per-SM idle >= 44 us inside the T=256 expert window". UFS is t256-designs.json angle 1: one persistent,
dependency-aware launch after the gather. It has one build_map, one fix-up round and per-expert w13 -> w2 readiness
counters, and hot items stay on wgmma.

**Verdict: CLOSE UFS.**
- Judge-scaled fillable idle in the expert window is **~11-12 us**, 14 us at most. The bars were build-go >= 44 and
  stack-only 20-44.
- Even with the head and tail that UFS does not touch (align, gather, combine) added in, it is ~18 us.
- 12 us on today's 254.7 us row is **~0.8% headline**.

- **Everything ran** as the operator on **gpunode06 GPU 1 only**, through guarded `bin/hopper-probe run`. There was no Slurm
  allocation and only node-local `/tmp/USER-hopper`, unstaged after each run. Nothing was refused, the guard never
  fired, and nothing is left running.
  - SM clock: 1830 MHz idle and 1803 MHz mean inside the window (GPC samples). Max 36 C, no throttle.
  - The nsys reports were reduced on the node and deleted there. Only the reduced JSON came back, to the takemyjob
    worktree, not the workstation's /tmp.
- **Code:** takemyjob branch **`hopper/ufsrung-1030` de0a41b** (worktree `~/git/.takemyjob-wt/ufsrung`, off
  `hopper/mstack-1030`):
  - `probes/ufsrung/force.py`: the node-only judge-route force;
  - `force_instr.py`: per-CTA instrumentation;
  - `mkcfg.py`: writes the configs `ufsr-r1/r2/r3/i1`;
  - `anat.py`: the window anatomy from the reduced trace.
  - `nj_profile.py` gained a `profile.gpu_metrics` option. It adds `--gpu-metrics-devices=cuda-visible
    --gpu-metrics-frequency=<max from nsys's help>` to the E3 capture and reduces `GPU_METRICS` per profiled rep
    window to `prof/gpumetrics.json.gz`.
  - Bundles `njufsr` and `njufsr2`.
- **Bytes:** master `muse_moe_w4a16/submission.py` = c5e2d8d4 = scored 45ef593d (2.0793), sha256 6b99a278..., 981,562 B.
  - Forced snapshot: `snap/ufsr-887ee121894a.py`.
  - Instrumented snapshot: `snap/ufsri-1d29e3a8007b.py`.
  - Both are untracked; rebuild them with force.py / force_instr.py.

| run (`runs/…-gpunode06`) | config | what |
|---|---|---|
| `20261029-195104` | ufsr-r1 | forced judge route; 40 timed + 40 profiled reps; nsys graph-node trace + GPU metrics at 50 kHz (the help-text regex missed the quoted max) |
| `20261029-195505` | ufsr-r2 | the same at **200 kHz** (5 us samples), another seed |
| `20261029-195951` | ufsr-r3 | r2 + E1 arm `nocp` (post `ms.cp.256 = 0`: SGLang post-norm instead of the PDL `ea_combine`), both profiled on identical inputs |
| `20261029-200325` | ufsr-i1 | instrumented snapshot, no profiler: per-CTA `%globaltimer` / `%smid` for every ea4x w13/w2 and wg w13/w2 CTA, 40 judge-style draws x 3 replays |

All runs: cold builds, passed=True, 5/5 TASK tests. Prepare took 143-145 s (budget reported, not enforced). Profiled vs
unprofiled row fidelity was x1.0033 / x1.0117 / x1.0072 (tol 3%). Node row (median, overhead removed):
- uninstrumented: 304.0 / 301.1 / 301.3 us;
- instrumented: 298.7 us, so the instrumentation does not slow the row.

## 1. Route: the judge's, forced where the node races differently

From the judge's T=256 PATH lines in the 45ef593d log:
- hybrid **H-16f** (fork);
- cold half on the **v4c twin v4cxs**, with [ea-enc] **E4XL** (cold_e 5);
- hot half **wh5ds** `wg_kernel<G,14>`;
- [mstack] **alcp** (`ms_align` + `ea_combine` pdl 1).

Left alone, the node routes Engine A T=256 to MASTER (so the cold half takes the v4 twin v4xs) and keeps [ea-enc]
master at 256 (HBM wall). In one mstack-ab party it picked wh over wh5ds.

force.py is appended to the bytes (node-only). It never changes a kernel:
- sets `ea.on[256] = 'v4c'` before `_hm_setup`, which is the judge's Engine A T=256 route;
- re-asserts `cold_x[256] = 'v4cxs'` after `_hmx_cold`, if that arm is canary-clean. It never had to fire: the node's
  own race picked v4cxs at 1.016;
- `EAE_T = (256,)` and EAE_BAR / XL_TIE 1.10, with only E4XL admitted at T=256. Exactness is still required, and it
  passed;
- WH_5VSWH and MS_BAR 1.10.

Every run printed `[ufsrung] ROUTE T=256 {'on': -16, 'fon': True, 'cold_bits': 112, 'cold_x': 'v4cxs', 'cold_e': 5,
'whk': 14, 'al': 1, 'cp': 1} -> JUDGE ROUTE`. The hybrid and [engineA] copy checks passed, and so did the [ea-enc] and
[mstack] state-copy checks.

## 2. Window anatomy (node)

Kernel edges, from the r3 board trace (us from `ms_align` start, medians over 40 reps). The per-CTA run (i1) agrees to
~2 us.

| kernel | grid | start -> end | note |
|---|---|---|---|
| ms_align | 33 | 0 -> 4.5 | 33 CTAs: SMs Active 28% |
| wg_gather | 2528 | 4.8 -> 8.3 | |
| cold w13 (ea4x E4XL) | 264 (2/SM, one wave) | 7.2 -> 149 | takes every SM; the hot w13 cannot start until whole SMs free up |
| hot w13 (wh5ds) | 132 (1/SM) | 121.5 -> 201 | starts in cold w13's tail |
| cold w2 | 264 | 179 -> 273 | starts in hot w13's tail |
| hot w2 | 132 | 253 -> 297 | starts in cold w2's tail and **ends last** |
| ea_combine (PDL) | 256 | 278 -> 302 | resident from 278, waiting ~17 us on the join |

So the fork runs as a four-phase relay: C13 -> H13 -> C2 -> H2. Each phase's tail is already filled by the next
kernel's ramp. The halves are zero-sum per SM (t256part, m03c).

### 2a. SMs-Active idle (nsys GPU metrics, 200 kHz, r2/r3)

idle = ∫(1 - SMs Active) dt, in us per SM. Samples are piecewise constant and stamped at the start of their 5 us period:
`ms_align` then reads 28-29% (33/132 = 25%), while end-stamping smears the flush into it and reads 82%. r3 board /
r2 / nocp:

| state (resident kernels) | t us | idle us/SM | SMA % | UFS-fillable? |
|---|---|---|---|---|
| align | 4.5 | 3.25 / 3.41 / 3.26 | 28 | no (before the gather) |
| gather (+ C13 ramp, gap) | 3.8 | 0.82 / 0.93 / 0.89 | 72-92 | no |
| C13, C13+H13, H13, H13+C2, C2, C2+H2 (the relay) | 264 | **0.45 / 0.47 / 0.70** | 99.3-100 | yes, but it is ~0 |
| H2 (+ CB resident) | 17-23 | 0.15 / 0.15 / **8.03** | 99 / 99 / **62** | yes |
| CB alone / post-norm | 5.6 / 10.1 | 2.13 / 2.20 / 2.70 | 62 / 73 | no (after the join) |
| **total** | 301-306 | **6.8 / 7.2 / 15.6** | | |

**Masking:** `ea_combine`'s PDL launch keeps its CTAs resident during the hot w2 tail, so SMs Active reads 99% there
and the board's total (6.8-7.2 us) hides the tail. The `nocp` arm, whose post-norm is not PDL, exposes it as 22.8 us of
hot-w2-only time at 62% SMs Active = **8.0 us/SM**. That idle is real on the board too: its SMs hold only waiting
combine CTAs. nocp is x0.9826 [0.9803, 0.9855] slower overall, which is cp's known gain.

### 2b. Per-CTA capacity idle (i1: every expert CTA's SM, start and end; 120 replays)

`%globaltimer` steps are 32 ns. CTA counts are 264 / 264 / 132 / 132.

SM capacity: an ea4x CTA counts as 0.5 SM (2 per SM) and a wg CTA as 1.0 (a whole SM). The window runs from the first
C13 CTA to the last H2 CTA, mean 293.7 us. It excludes align, gather and combine, which are outside UFS.

| phase (by kernel edges) | dur us | empty-SM idle us/SM | half-occupied us/SM (x0.5) | what it is |
|---|---|---|---|---|
| H2 alone | 23.2 | **7.76** | 0 | hot w2's last wave: static stride, staggered ends |
| C13+H13 hand-off | 27.7 | 0.48 | 3.21 (**1.60**) | an SM keeps one cold CTA, so a wg CTA cannot land there (head-of-line) |
| C2+H2 hand-off | 21.5 | 0.88 | 1.99 (**0.99**) | the same at the cold w2 tail |
| C13 | 118.3 | 0.41 | 0.47 (0.24) | launch ramp |
| C2, H13, H13+C2, C13+H13+C2 | 107 | 0.36 | 0.28 (0.14) | the H13 -> C2 hand-off is clean: a freed wg SM takes two ea4x CTAs |
| **expert window** | 293.7 | **9.90** | 5.94 (**2.97**) | **capacity idle 12.86 us/SM** (15.84 if a lone ea4x CTA were worth nothing) |

Per replay, capacity idle is median 12.6 us, range 6.5-22.4, p90 16.5. The H2-alone part is median 7.6 (2.6-16.5).
Half-occupied is 4.4-8.1.

The two methods agree:
- the H2 tail is 8.0 us (GPU metrics, unmasked) vs 7.76 us (per-CTA);
- the relay is ~0.5 us of empty SMs in both.

What GPU metrics cannot see is the half-SM head-of-line time, ~3 us/SM. The design panel predicted 10-15 us for it;
measured, it is 3.

There are no PDL hand-off drains to remove:
- C2 and H2 start 30-50 us after their predecessors on the same stream end, so neither ever waits in
  griddepcontrol.wait. Only the combine waits.
- Launch gaps are 0.28 us in all.

## 3. Judge scaling

Each node idle piece is scaled by the judge/node ratio of the kernel whose tail it is. Both GPUs have 132 SMs, and the
judge's SM clock under load is ~1769 MHz vs the node's 1803 in the window.
- **Hot-tail pieces** (H2 alone, C2+H2 empty): x **0.956**. This is the wh5ds hot pair, judge 93.1 us (45ef593d
  [hh-wh]) vs node 97.4 (r1). The hot half is compute-bound, so this is close to the clock ratio.
- **Cold-tail pieces** (half-SM HOL, the C13 ramp): x **0.73-0.88**.
  - That is judge/node cold kernel duration: C13 104.4 / ~142, C2 81.4 / ~92, from the 45ef593d profile vs r3.
  - The bounds bracket the HBM ratio, 3.11 measured / 4.37 ring = 0.71. Cold CTA durations, and with them their tail
    skew, are HBM-bound on the node.

| piece | node us/SM | factor | judge-scaled |
|---|---|---|---|
| hot w2 tail (H2 alone 7.76 + C2+H2 empty 0.88) | 8.64 | 0.956 | **8.26** |
| half-SM head-of-line at 0.5 (C13+H13 1.60, C2+H2 0.99, C13 0.24, C2 0.13) | 2.96 | 0.73-0.88 | **2.2-2.6** |
| other empty SM time in the relay (C13 0.41, C13+H13 0.48, H13+C2 0.29, H13 0.05, C2 0.02) | 1.25 | 0.73-0.96 | **0.9-1.2** |
| **UFS-fillable, judge-scaled** | **12.86** | | **11.3-12.1 us** |
| upper bound (a lone ea4x CTA counted as a fully idle SM) | 15.84 | | ~14.0 |
| not UFS, for completeness: align + gather SMA idle (4.1 node; judge ms_align 4.1 + gather 3.5 vs node 4.5 + 3.5) | 4.1 | ~0.95 | 3.9 |
| not UFS: combine alone (2.1; judge ea_combine iso 5.09 vs node 5.12) and launch gaps (0.28) | 2.4 | ~1.0 | 2.4 |
| **everything idle in the row** | | | **~18 us** |

The judge may sit lower still. m256's t256anat reading was "the hot w2 tail is covered by the cold w2 under the fork on
the judge (0 us uncovered)". If that holds, the 8.3 us hot-tail piece shrinks there, so the scaled figure is an upper
estimate.

Sizing on today's row (judge timed median 254.7 us, 45ef593d), with headline = (row / (row - d))^(1/6):
- d = 12 us gives **0.81%**; d = 14 gives 0.95%;
- the 20 us stack-only floor would be 1.37%;
- the 44 us rung would be 3.2%. At 254.7 us the 3% point is now d = 41.4 us, and the pre-registered 44 stands.

## 4. Verdict against the pre-registered rule

**Close UFS: judge-scaled fillable idle ~11-12 us (at most ~14 us) < 20 us.**
- Even adding the idle UFS cannot touch (align, gather, combine, gaps), the whole row holds ~18 us of idle SM time.
- On the node, the relay between the four expert kernels is packed. Empty SM time there is ~0.5 us, and half-SM
  head-of-line time is ~3 us per SM.
- The only sizeable idle is the hot w2 last-wave tail, 7.6-8.3 us. That is a hot-half tail lever (wg2 stream-K / LPT
  tail work), not a reason for a unified persistent schedule. The single-kernel wg2 tail forms are already [refuted] in
  CLAIMS, and their reopen condition is ">= 4 us uncovered hot tail" on the judge, which t256anat put at 0.

**What this does not cover.** UFS was also sized for removing busy fixed cost, not idle: one of the two build_maps,
one ring fill, one fix-up round (judge D +4.1 / +5.0 us) and the K1 -> K2 hand-off. That busy cost is part of the
27.4 us cold fixed term in m256's anatomy. The rung measures idle, so it is not counted here.
- Generously, removing it is ~-8..-12 us. With the idle that gives ~20-24 us, ~1.4-1.6%. That is m256's "fused
  persistent cold engine" line (-8..-15 us), which already sits on the shelf below the bar.
- It only helps inside a multi-build stack with the shelf riders (xlres C ~+0.2%, hhxl ~+0.2%).
- It is not a reopen of UFS. The design's ceiling arguments (-41 us) assumed 10-15 us of head-of-line plus a large
  packable relay, and both measured small.

Reopen conditions (any one):
- a judge trace (profiler rows with start times, not durations) showing >= 20 us of uncovered hot w2 tail or relay idle
  on the H200;
- a cold engine whose fixed cost alone measures >= 25 us removable on the judge (consume-only K1+K2 chain at T=256).

## 5. Method notes for reuse

- `profile.gpu_metrics` in nj_profile works unprivileged on gpunode06 (nsys 2026.4.1, gh100 set, 30 metrics, 200 kHz).
  DCGM was left running and nothing changed node state. Metrics are stamped at the start of each period.
- **A PDL-launched tail kernel masks SMs Active.** Its resident, waiting CTAs read as active. Unmask it with an arm
  whose tail kernel is not PDL, or use per-CTA records.
- The per-CTA recorder (`force_instr.py`) needs no extra kernel arguments:
  - a `__device__` buffer, a RAII `UfsrRec` (thread 0 stamps `%globaltimer`/`%smid` at entry; each warp's lane 0 does
    `atomicMax` at exit);
  - a `ufsr_dump` host function added to the module's function list;
  - a driver appended to `prepare()` that replays kernel_fn graphs with a read flush.
  - Registers stayed within the [ea-enc] XL admission (93-94 / lmem 0 / occ 2), and wh5ds kept r168 / l0 with
    ptxas -Werror clean.
