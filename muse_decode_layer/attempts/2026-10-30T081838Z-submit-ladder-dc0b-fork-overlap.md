# muse_decode_layer · 2026-10-30T081838Z · submit · ladder-dc0b-fork-overlap

Hypothesis: `muse_decode_layer/hypotheses/queue/2026-10-30T052308Z-ladder-dc0b-fork-overlap.md` (branch `ladder/dece2`).

- **Job:** `43265863258a47a0` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `5e4601ed` (ladder/dece2), 995,734 B, sha256 b73628f6b0a8...
  Base: 64a6861f (the scored compliant board 44de5103) + 7a38ece1 (fork arm) + 5e4601ed (r32 arm).

## Goal
Win back DC0's router cost (judge 7f5e68e6: d1 +~9 us, v1 +24-28 us vs the old BF16 router) while staying exactly
compliant. Root cause measured on the node (runs/20261029-223422..223940-gpunode06 r32bench): cuBLAS F.linear FP32 at
m=8 x 256 x 8192 takes 21.5 us (m=1: 6.1 us), far above its 8.4 MB byte floor.

## What changed, and where
**Mechanism:** the T <= 8 FP32 router (FP32 x from SGLang's boundary, FP32 router weight, FP32 accumulation) raced in
prepare per T in {1, 8} over 4 arms: serial F.linear (DC0), F.linear on the fork stream overlapping the pooler, a
split-K FP32 GEMV (`_dc0_r32_kernel`, Triton, CUDA-core FP32 FMA, fixed-order partial sum: deterministic) serial, and
the GEMV on the fork stream. The GEMV is admitted only if max |r32 - F.linear| <= 1e-5 on a probe input and it repeats
bitwise; fork arms must be bitwise equal to their serial twin. Winner by `_rt_select` margin 1.01, else serial F.linear.
T = 2..7 and T > 8 unchanged. `[dc0b-fork] PATH CHOSEN T=<t>` lines name the pick.
Node (gpunode06 GPU 6, runs/20261030-002337-gpunode06): race picked R32FORK at both T: T=1 layer serial/r32fork x1.025-1.027,
T=8 x1.075 (serial/fork x1.033-1.037, serial/r32 x1.040-1.043). dc0gate (kernel_fn hooked, judge reference inputs):
5,296 token rows over the prepared routes + forced fork-all + forced r32-all passes: router input float32 and unrounded
in every call; consumed logits vs F.linear fp32 <= 7.6e-6 (r32) / 0 (F.linear arms); top-8 identical 5,296/5,296;
0 check failures -> PASS. nodejudge dc0b2-val (cold, budget enforced): prepare 121.4 s, 13/13, 12 shapes x 2 checked.

## Rules check
- [x] **Router precision (rules :107-108, :137): at T <= 8 the router consumes SGLang's unrounded FP32 router input and
      the FP32 router weight with FP32 accumulation in every arm (F.linear, or the FP32 split-K GEMV whose logits are
      within 7.6e-6 of F.linear, the same size as F.linear's own error vs FP64, top-8 identical); T > 8 the BF16 hi + lo
      weight pair with FP32 accumulation, as SGLang above 8 tokens. No path consumes a BF16-rounded router input at T <= 8.**
- [x] w2 split-K reduce stays pinned to SGLang's FP32 path (64a6861f).
- [x] CUDA-graph capturable: stream fork/join and Triton launches captured; no sync / .item() in kernel_fn; choices in prepare.
- [x] Nothing cached across calls; weights unchanged; activations BF16 where SGLang keeps BF16; FP8 KV unchanged.
- [x] No timing patches, harness access, reference import, subprocess or network.
- [x] Size 995,734 B < 1,000,000.

## Risks
low: the race may pick serial on the judge (then = DC0). Expected: v1 rows -8..-15 us, d1 rows -1..-3 us vs DC0
(~+1.5-3% headline), still above the old non-compliant router cost.


## Why a scored run
Two checks of the DC0b router race beat the board 1.671: b734431f 1.684 (DC0b alone, these bytes 5e4601ed) and 50d63183 1.682 (with the null DC1). v1 rows -20 us vs DC0. Compliance unchanged (FP32 x, FP32 W, FP32 accumulation at T<=8). Standing rule: scored submit when a check beats the board.

## Result (after the run)
- job `43265863258a47a0` (submit), passed, **1.690x [1.665, 1.716]** vs board 1.671 -> landed on master (bytes 5e4601ed).
