---
problem: muse_decode_layer
slug: dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32
created: 2026-10-25T153205Z
base: master
area: qkv GEMM + o GEMM + unpooler GEMM (T <= 32)
change: >
  Port muse_bf16_proj master's swap-AB wgmma GEMMs (commit 040fc7e18f, best 1.2491x submit
  ee4f89f2f62c42d3) to decode's qkv_gate and o GEMMs at every T <= 32, and to the full-width
  unpooler at 9 <= T <= 32. bf16_proj measured these exact weight shapes on the same judge with
  one launch each (no splitKreduce node): qkv_gate 9.7 us at m=1/8, 10.2 at m=16, 11.5 at m=32
  (decode master: nvjet 128x8/64x16/64x32 splitK 12.2-13.1 us PLUS a splitKreduce 1.8 us node); o
  3.9 / 4.3 / 4.4 / 4.7 us at m=1/8/16/32 (decode master: nvjet 64x8 6.8-7.1 us at T=1/8; at
  T=16/32 the 2-call nvjet row 26.4-27.4 us is o + unpooler). Copy verbatim from
  muse_bf16_proj/submission.py on master: SWAPAB_* and SWAPAB16_* constants and configs (the m <=
  16 BLOCK_T=16 path and the 17 <= m <= 32 BLOCK_T=32 path), _bf16_swapab_kernel and its m<=16
  sibling, _swapab_route_ok, _swapab_forward, _swapab_buffers, _swapab_prepare (per-state
  workspaces and self-resetting tickets; warm every m in 1..32 per layer on synthetic bf16 inputs,
  synchronize, zero the tickets), and the OutOfResources stage fallback. Only the layers qkv_gate
  (2304 x 8192), o (8192 x 1024) and unpooler (8192 x 4096, full width at T > 8) are routed; the
  pooler and router stay inside dense-chain kernel P at T <= 32 (so the held
  bf16-m32-swapab-wgmma-port's concern, the swap-AB pooler competing with the side-stream router
  for SMs, no longer exists: there is no side stream at T <= 32 on master). Call sites in
  _parent_attention: qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o, dense['o']) ->
  _swapab_forward when tokens <= 32 and this state's per-(layer, T bucket) flag is set; in
  kernel_fn the T > 8 unpooler _gemm likewise at 9 <= T <= 32. Full decode_1 with dl3 piece A on
  keeps piece A (its own Q/O kernels); T <= 8 unpooler keeps dl3 piece C / the slice GEMM.
  SELECTOR per (layer, T in {1, 8, 16, 32}): the bf16_proj model selector (graph replay, 256 MB
  flush, 41 interleaved reps, empty-graph baseline subtracted, >= 1.02x win, assert enabled ==
  printed winner), timing the GEMM INCLUDING master's splitKreduce node; print '[swapab] PATH
  CHOSEN <layer> T=<t>: swapab X us vs cuBLAS Y us'. Self-check rel-L2 vs an fp32 torch.mm
  reference at every m in 1..32, bound 1e-2, on two alternating states. This fixes the hold
  reasons of bf16-m32-swapab-wgmma-port (now one combined restack of the decode GEMMs with
  per-cell attribution, pooler excluded) and ports the newer bf16 master kernel rather than
  re-running qkv-o-swapab16-t2-t16-no-fork's code (1.3056x, unproven-vs-best; it also covered
  T=2..16 only). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no
  timing, no prints, no try/except, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow and no per-call mutable Python state; gates read config/shape
  and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4
  stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key
  attended; no child processes. The judge prepares TWO state copies and rotates them: keep every
  device pointer, TMA map, ticket array and workspace per state, never in a module global. Every
  new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode
  arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an
  in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any
  miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check
  by capturing once and replaying with refilled inputs while alternating two states. Selectors
  time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps,
  empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS
  by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per
  module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First
  measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: qkv nvjet 12.2 + splitKreduce 1.8 -> ~9.7 us; o 7.1 -> ~3.9 us; ~-7 us on 117.8 us (weight 46)
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: qkv 12.6 + reduce ~1.7 -> 9.7; o 6.9-7.1 -> 4.3; ~-6 us on 190/199 us"
  - shapes: "*load=decode_16*"
    direction: faster
    why: qkv 12.8 + reduce -> 10.2; o + unpooler 2-call row 26.4 -> ~4.4 + unpooler swapab; ~-4..-7 us
  - shapes: "*load=decode_32*"
    direction: faster
    why: "qkv 13.1 + reduce -> 11.5; o 4.7; unpooler bf16 m32 18.5 vs nvjet share of the 26.9 us row; ~-3..-6 us on 236/442 us (sliding decode_32 is master's worst ratio, 1.075x)"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: piece A owns qkv/o there when on; otherwise the same -7 us as sliding
  - shapes: "*load=verify_16*"
    direction: flat
    why: "T=128: untouched"
  - shapes: "*load=verify_32*"
    direction: flat
    why: "T=256: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T162203Z
review_companions: null
hold_reason: "adversarial review: Hold. The conjecture is that bf16 master's 'newer' swap-AB kernel will run in decode at its standalone 9.7/3.9 us. That premise is false, and the transfer has a"
---

# dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32

## Change

Port muse_bf16_proj master's swap-AB wgmma GEMMs (commit 040fc7e18f, best 1.2491x submit ee4f89f2f62c42d3) to decode's qkv_gate and o GEMMs at every T <= 32, and to the full-width unpooler at 9 <= T <= 32. bf16_proj measured these exact weight shapes on the same judge with one launch each (no splitKreduce node): qkv_gate 9.7 us at m=1/8, 10.2 at m=16, 11.5 at m=32 (decode master: nvjet 128x8/64x16/64x32 splitK 12.2-13.1 us PLUS a splitKreduce 1.8 us node); o 3.9 / 4.3 / 4.4 / 4.7 us at m=1/8/16/32 (decode master: nvjet 64x8 6.8-7.1 us at T=1/8; at T=16/32 the 2-call nvjet row 26.4-27.4 us is o + unpooler). Copy verbatim from muse_bf16_proj/submission.py on master: SWAPAB_* and SWAPAB16_* constants and configs (the m <= 16 BLOCK_T=16 path and the 17 <= m <= 32 BLOCK_T=32 path), _bf16_swapab_kernel and its m<=16 sibling, _swapab_route_ok, _swapab_forward, _swapab_buffers, _swapab_prepare (per-state workspaces and self-resetting tickets; warm every m in 1..32 per layer on synthetic bf16 inputs, synchronize, zero the tickets), and the OutOfResources stage fallback. Only the layers qkv_gate (2304 x 8192), o (8192 x 1024) and unpooler (8192 x 4096, full width at T > 8) are routed; the pooler and router stay inside dense-chain kernel P at T <= 32 (so the held bf16-m32-swapab-wgmma-port's concern, the swap-AB pooler competing with the side-stream router for SMs, no longer exists: there is no side stream at T <= 32 on master). Call sites in _parent_attention: qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o, dense['o']) -> _swapab_forward when tokens <= 32 and this state's per-(layer, T bucket) flag is set; in kernel_fn the T > 8 unpooler _gemm likewise at 9 <= T <= 32. Full decode_1 with dl3 piece A on keeps piece A (its own Q/O kernels); T <= 8 unpooler keeps dl3 piece C / the slice GEMM. SELECTOR per (layer, T in {1, 8, 16, 32}): the bf16_proj model selector (graph replay, 256 MB flush, 41 interleaved reps, empty-graph baseline subtracted, >= 1.02x win, assert enabled == printed winner), timing the GEMM INCLUDING master's splitKreduce node; print '[swapab] PATH CHOSEN <layer> T=<t>: swapab X us vs cuBLAS Y us'. Self-check rel-L2 vs an fp32 torch.mm reference at every m in 1..32, bound 1e-2, on two alternating states. This fixes the hold reasons of bf16-m32-swapab-wgmma-port (now one combined restack of the decode GEMMs with per-cell attribution, pooler excluded) and ports the newer bf16 master kernel rather than re-running qkv-o-swapab16-t2-t16-no-fork's code (1.3056x, unproven-vs-best; it also covered T=2..16 only). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_1 *` faster: qkv nvjet 12.2 + splitKreduce 1.8 -> ~9.7 us; o 7.1 -> ~3.9 us; ~-7 us on 117.8 us (weight 46)
- `*load=verify_1 *` faster: T=8: qkv 12.6 + reduce ~1.7 -> 9.7; o 6.9-7.1 -> 4.3; ~-6 us on 190/199 us
- `*load=decode_16*` faster: qkv 12.8 + reduce -> 10.2; o + unpooler 2-call row 26.4 -> ~4.4 + unpooler swapab; ~-4..-7 us
- `*load=decode_32*` faster: qkv 13.1 + reduce -> 11.5; o 4.7; unpooler bf16 m32 18.5 vs nvjet share of the 26.9 us row; ~-3..-6 us on 236/442 us (sliding decode_32 is master's worst ratio, 1.075x)
- `*layer=full load=decode_1 *` flat: piece A owns qkv/o there when on; otherwise the same -7 us as sliding
- `*load=verify_16*` flat: T=128: untouched
- `*load=verify_32*` flat: T=256: untouched

## Risk

In-layer interference: bf16_proj timed each GEMM alone; in decode the qkv GEMM follows the ffn boundary and the o GEMM follows attention, both on one stream, so standalone gains should carry, but LESSONS records a T=8/16 loss for Triton split-K GEMMs in decode (bf16-shape-gated-triton-gemms-small-t, a different kernel family); the per-cell in-prepare selector is the guard. Read the swapab rows and the missing splitKreduce rows in the profile per shape, not the headline. Overlaps dl3-unpool-nf1-t2-t8 (queued) only in that both touch the unpooler; this one routes the unpooler at T >= 9 only, so they are disjoint.

## Adversarial review

Reviewed 2026-10-25T162203Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. The conjecture is that bf16 master's 'newer' swap-AB kernel will run in decode at its standalone 9.7/3.9 us. That premise is false, and the transfer has already been measured: the qkv/o kernel and configs at m<=16 are byte-identical to what qkv-o-swapab16-t2-t16-no-fork ran (job b0057d2e6cef4d5a), where the same kernels measured qkv 12.7-13.0 us and o 6.2-6.5 us in context, and sliding verify_1/decode_16 were flat. The genuinely new cells (sliding decode_1 T=1, T=32, unpooler T=16/32) project to about -1..-2.5 us per shape (~0.3-0.6% geomean), which a scored submit cannot resolve; bf16_proj already measured the T=16 full unpooler cell as a regression.

### Justification
A. MECHANISM: swap-AB at m<=32 is real on H100/H200 (wgmma m64n16/n32 instead of mma.sync), and it removes cuBLAS's splitKreduce node (1.8 us in decode). bf16_proj does measure _bf16_swapab_kernel at qkv 9.5-9.8 us (m=1/8/16) and 11.6 (m=32), and o 3.9-4.7 us (pooler-nn-layout-widen-m256.result.json). BUT THE TRANSFER HAS ALREADY BEEN MEASURED, and it fails: decode's qkv-o-swapab16-t2-t16-no-fork ran the identical kernel. git diff b99b1dd3..master on muse_bf16_proj/submission.py touches only the unpooler-slice / ouns routing; SWAPAB16_CFG qkv (2304,8192,SK4,BK128,4st), o (8192,1024,SK1,BK256,4st), BLOCK_T=WS_M=16 and _bf16_swapab_kernel are unchanged, so the 'newer kernel rather than re-running qkv-o-swapab16's code' claim is false at T<=16. Its profile (archive/qkv-o-swapab16-t2-t16-no-fork/b0057d2e6cef4d5a.profile.txt) shows _swapab_qkv_kernel at 12.7-13.0 us vs nvjet 128x8/64x16 splitK at 12.0-12.6 + splitKreduce 1.8 us, a saving of ~1-1.5 us, not ~4.5. _swapab_o_kernel ran at 6.2-6.5 us vs nvjet 64x8 at 6.6-7.0, a saving of ~0.5 us, not ~3. Decode's own prepare-time graph selector in that run (log lines 119/180) saw qkv T8+T16 40.6 vs 43.3 us (1.066x) and o 24.7 vs 24.8 (1.004x, which would be OFF under this hypothesis's 1.02x rule). Result: sliding verify_1 191.58 vs 191.42 (flat), sliding decode_16 204.72 vs 204.68 (flat). The Triton kernel loses ~3 us (qkv) and ~2.3 us (o) in decode relative to bf16_proj's rows, while nvjet does not. The cause is unidentified, but it is reproducible. PER-CELL REALISTIC EXPECTATION with that context penalty: sliding decode_1 T=1 qkv ~12.7 vs 13.8 (nvjet+reduce), o ~6.3 vs 7.0, so ~-1.5..-2 us on 118 us (~1.5%, inside the ~1.5% noise). T=8/16: already measured flat (duplicate). T=32: qkv ~14 vs 12.2-12.6+1.8, o ~7 vs its share of the 26.4-27.7 us 2-call row, so ~-1 us on 239/431 us (0.2-0.4%). Unpooler T=32: bf16 itself shows 18.8 swap-AB vs ~18.5 nvjet (no win even standalone). Unpooler T=16: bf16_proj ran exactly this cell (unpooler-full-m16-swapab-route, done/regression: 18.51 vs 18.48 flat, others lost), and bf16 master deliberately keeps cuBLAS for the full unpooler at 9..16. So 'copy verbatim from master' and 'route the unpooler at 9..16' contradict each other. The 'expected' block (-7 us decode_1, -6 us verify_1, -4..-7 decode_16, -3..-6 decode_32) overstates the measured transfer by roughly 3-5x. WEAKEST LINK: the selector is the hypothesis's only guard against in-context loss, and it times the GEMM in isolation (graph replay + 256 MB flush), which is close to bf16_proj's conditions. It therefore cannot see the context penalty that sank the earlier port: its qkv cell would be ON again for the same flat shape result. CONFIRM/REFUTE: a scored run cannot read a ~0.3-0.6% geomean move. The kernel rows can be read, but for T<=16 they are already known.

### Confounders on the current kernel
- qkv-o-swapab16-t2-t16-no-fork (not landed, neutral/unproven-vs-best, job b0057d2e6cef4d5a): same kernel, same qkv/o configs, same call sites at T=2..16. It is the direct prior measurement, and this run would re-measure it at T=8/16 (identical code at those cells).
- dense-chain-s3-t1-q-sk2-o-route-unpool + dl3-piece-a-gate-stride-check-fix: piece A owns qkv/o at full decode_1, and piece C owns the unpooler at T<=8. If piece A falls back at prepare, as it did in b0057d2e's log, the full decode_1 cell moves too, so read full decode_1 only together with the [dl3] PATH CHOSEN line. Otherwise this run does not confound them; the gates are disjoint by design.
- small-t-launch-minimal-dense-chain-pdl: P/R run as a PDL chain at T<=32 right after the attention boundary. A non-PDL Triton o kernel ahead of that chain can cost the overlap nvjet gives. This is one plausible part of the measured ~2-3 us in-context penalty, but unverified.
- full-decode-sk-ticket-restack / splitkv: these produce o at full decode_16/32 as a reshape of the sk output. The layout is warmed in prepare in the earlier port. No new interaction beyond the one above.

### Interference and A/B needs
No companion would rescue interpretability. The binding problem is not interference with a landed change. The effect of the new cells (~1-2 us on 118-431 us shapes) is below per-shape noise, and the T<=16 cells repeat a measured-flat result. A control (e.g. a qkv-only or T=1-only cell) would cost a second scored run to read a sub-noise effect. If this is re-minted, the right fix is in-prepare rather than a companion. (1) Time each cell IN CONTEXT: capture kernel_fn (or _parent_attention plus the following boundary/P launch) for that T with the flag on and off, interleave the replays, and gate on >=1.02x of the whole captured segment, not the isolated GEMM. That is the only selector that can see the penalty seen in b0057d2e. (2) Restrict to cells not yet measured: sliding decode_1 T=1 qkv/o and T=32 qkv/o. Drop T=8/16 (duplicate), drop unpooler T=16 (bf16 regression) and unpooler T=32 (no standalone win). (3) Fix the expected block to -1..-2 us per shape. That version is best folded into a combination round rather than spending a standalone scored submit.

### Open questions
- Why does the identical _bf16_swapab_kernel run ~3 us slower in decode (12.7-13.0 vs 9.5-9.8 for qkv; 6.2-6.5 vs 3.9-4.2 for o) while nvjet stays near its standalone time? Candidates: bf16_proj's profile leaves the 37.7 MB qkv weight partly L2-resident across reps (decode's preceding kernels evict it); loss of PDL/launch overlap next to the dense chain; or SM clocks under the layer's Marlin/attention load. Until this is explained, no bf16_proj Triton GEMM row should be used as a decode expectation.
- Does decode's sliding decode_1 still run nvjet 128x8 splitK + splitKreduce for qkv on current master 6773ec5c (the BEST profile's top-3 no longer shows it)? If yes, T=1 sliding is the one untested cell worth ~1.5-2 us.
- If a re-mint is wanted, should it be the T=1-sliding + T=32 qkv-only slice, gated by an in-context whole-segment selector, folded into a combination submit?

