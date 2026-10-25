---
problem: muse_decode_layer
slug: combo-attn-master-sliding-race-skw-c16
created: 2026-10-25T161651Z
base: master
area: "combination: sliding attention per-band race (B=1, 2..16, 17..32, verify B=1) + routing R / unpooler tail"
change: >
  COMBINATION (conjectural, cross-problem): make decode's whole sliding attention side a per-band
  race between decode's measured winner and muse_attention_decode master's routes (commit
  7e12891512, 1.5017x check 252abb7f11ec48c3), on master 6773ec5cf5. Includes the C1 tail so the
  race is the only difference on its rows. SOURCE A1 attention sk-window (attention hypotheses
  restack-sliding-decode-sk-window-on-cluster-master and
  restack-sk-pseudo-cluster-plus-v3-fused-preattn; same port as queued
  sliding-sk-window-port-decode32-verify1): build attention master's sk CUDA TU as a NEW module
  muse_decode_sk_window_v1 beside decode's _SK (which stays byte-identical for full decode); copy
  _pre_attention_kernel (PAIRED=False, RoPE, k/v FP8 into the cache, kv_scale descale), the
  variant table and the PSEUDO prefix loader verbatim; variant 6 (PSEUDO TOK=1 single-level
  ticket) for sliding decode and variant 5 (PSEUDO TOK=8 two-level) for sliding verify B=1; output
  is the normed, gated bf16 [T, 1024] o-GEMM input (gate at SK_QKV_GATE_OFF, assert the qkv row
  layout). In attention: sliding decode_32 23.85 -> 17.38 us shape (sk 14.0 + pre 1.9),
  head-to-head 19.5 us net vs cluster C=4 22.1 vs Triton 25.8; sliding verify_1 13.2 us vs SGLang
  22.6. SOURCE A2 attention cluster C=16 (attention stack-streamk-ticket-cluster-c16-c4, 13.15 ->
  12.57 us; master kernel row 10.8 us; same port as queued sliding-cluster-c16-decode1-port): new
  module muse_decode_slide_c16_v1 (non-portable cluster size attribute), sliding decode B=1 only,
  raced against decode's _sliding_fused_kernel (13.2 us in 933d). SOURCE D1 = C1's S1
  (sliding-fused-batched-decode-port 3ddcc4f2a8 / sliding-batched-triton-b16-band 1cabea713e,
  Triton S=16). SOURCE D2 = C1's S2 (stack-route-topk-pdl-unpool-nf1 2fe4d35e13: R variant 0 at 2
  <= T <= 32, unpool NF=1 at 2 <= T <= 8). SOURCE D3 = C1's S3
  (fused-preattention-paired-pad-restack 963bcf0267) for the FA3 feed of sliding verify B=1 only.
  THREE-WAY SELECTORS (one decision per band, all candidates timed in the SAME interleaved replay
  set; winner must beat master by >= 1.02x, else master): band SB1 sliding decode B=1 = {master
  _sliding_fused_kernel, A2 cluster C16}; band SB16 sliding decode 2 <= B <= 16 = {master FA3
  chain, D1 Triton S=16, A1 pre + sk variant 6}; band SB32 sliding decode 17 <= B <= 32 = {master
  FA3 chain, D1, A1}; band SV1 sliding verify B=1 = {master chain, D3 pre + FA3, A1 pre + sk
  variant 5}. Print each candidate's median and 'PATH CHOSEN: <band>: <winner>'. Checks: A1 and A2
  use attention master's acceptance rules (o rel-L2 vs an fp32 windowed reference with accurate
  sincosf rope; cache k within one E4M3 code by sign-magnitude ordinal on < 6% of bytes, v
  byte-exact, satfinite cvt accepted since sliding-cluster-c4-decode32-s2's probe showed torch
  saturates on this image), lens {1, 63, 2047, 2048, 2049, 80000}, ragged, kv_scale {1, 0.015625},
  positions to 2^20, two alternating states with refilled inputs; D1-D3 as in C1. D2 runs exactly
  as in C1 (its own gates). Full layers, T > 32 and the GEMMs are master byte for byte. HARD RULES
  (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no
  try/except, no allocation sized by input values (shape/config only), no data-dependent Python
  control flow and no per-call mutable Python state; gates read config/shape and this state's own
  prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant
  in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child
  processes. The judge prepares TWO state copies and rotates them: every device pointer,
  counter/ticket array, workspace and TMA map lives in the state, never in a module global (this
  includes the Triton sliding kernel's split counters: master's module-global _COUNTERS dict must
  NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per state and slice it).
  Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed load_inline module
  built at import like master's, with an in-process canary (synchronize + 'launching'/'ok' prints,
  flush=True) after each first launch; on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line
  keeps master for that cell. GATE PER SOURCE PER CLASS: each source path has its own prepare-time
  gate = (a) correctness check (bound stated per source below) on two alternating states with
  refilled inputs under captured-graph replay, then (b) a cold-L2 CUDA-graph selector (256 MB
  flush before each replay, >= 41 interleaved reps, empty-graph baseline subtracted) that enables
  the cell only when new/master <= 0.98 (i.e. master/new >= 1.02) and asserts the enabled flag
  equals the printed winner. Print exactly one 'PATH CHOSEN: <source> <class>: <winner> [new X us
  vs master Y us]' line per source per class. Every added prepare() phase runs AFTER master's own
  setup (after _dl3_finish), each with its own start deadline measured from prepare()'s t_prepare0
  (skip the phase with a fallback line if the deadline has passed) and a print of its own wall
  seconds plus the cumulative prepare wall time; master's own internal deadlines and its
  _sk_prepare budget do not move. Print compile seconds per load_inline module. First measurement
  is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "SB32: A1 expected winner (attention ranks it 6.3 us ahead of Triton); FA3 chain ~29.5 us of nodes -> pre+sk ~16 us; 239.2 -> ~222-226 us (vs C1's ~229)"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "SV1: A1 variant 5 replaces FA3 9.9 + combine 3.9 + prepare_varlen + ~5 small nodes with ~13 us; plus D2; ~-10..-14 us on 180"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "SB1: A2 cluster C16 10.8 us vs _sliding_fused_kernel 13.2 us; ~-2 us on 118"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "SB16: D1 or A1 (the race is the experiment); >= C1's ~-9 us"
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: D2 (unpool T=8 + R)
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: "full layers untouched except D2's R at T=16/32; control"
  - shapes: "*load=verify_16*"
    direction: flat
    why: untouched; control
  - shapes: "*load=verify_32*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-attn-master-sliding-race-skw-c16

## Change

COMBINATION (conjectural, cross-problem): make decode's whole sliding attention side a per-band race between decode's measured winner and muse_attention_decode master's routes (commit 7e12891512, 1.5017x check 252abb7f11ec48c3), on master 6773ec5cf5. Includes the C1 tail so the race is the only difference on its rows. SOURCE A1 attention sk-window (attention hypotheses restack-sliding-decode-sk-window-on-cluster-master and restack-sk-pseudo-cluster-plus-v3-fused-preattn; same port as queued sliding-sk-window-port-decode32-verify1): build attention master's sk CUDA TU as a NEW module muse_decode_sk_window_v1 beside decode's _SK (which stays byte-identical for full decode); copy _pre_attention_kernel (PAIRED=False, RoPE, k/v FP8 into the cache, kv_scale descale), the variant table and the PSEUDO prefix loader verbatim; variant 6 (PSEUDO TOK=1 single-level ticket) for sliding decode and variant 5 (PSEUDO TOK=8 two-level) for sliding verify B=1; output is the normed, gated bf16 [T, 1024] o-GEMM input (gate at SK_QKV_GATE_OFF, assert the qkv row layout). In attention: sliding decode_32 23.85 -> 17.38 us shape (sk 14.0 + pre 1.9), head-to-head 19.5 us net vs cluster C=4 22.1 vs Triton 25.8; sliding verify_1 13.2 us vs SGLang 22.6. SOURCE A2 attention cluster C=16 (attention stack-streamk-ticket-cluster-c16-c4, 13.15 -> 12.57 us; master kernel row 10.8 us; same port as queued sliding-cluster-c16-decode1-port): new module muse_decode_slide_c16_v1 (non-portable cluster size attribute), sliding decode B=1 only, raced against decode's _sliding_fused_kernel (13.2 us in 933d). SOURCE D1 = C1's S1 (sliding-fused-batched-decode-port 3ddcc4f2a8 / sliding-batched-triton-b16-band 1cabea713e, Triton S=16). SOURCE D2 = C1's S2 (stack-route-topk-pdl-unpool-nf1 2fe4d35e13: R variant 0 at 2 <= T <= 32, unpool NF=1 at 2 <= T <= 8). SOURCE D3 = C1's S3 (fused-preattention-paired-pad-restack 963bcf0267) for the FA3 feed of sliding verify B=1 only. THREE-WAY SELECTORS (one decision per band, all candidates timed in the SAME interleaved replay set; winner must beat master by >= 1.02x, else master): band SB1 sliding decode B=1 = {master _sliding_fused_kernel, A2 cluster C16}; band SB16 sliding decode 2 <= B <= 16 = {master FA3 chain, D1 Triton S=16, A1 pre + sk variant 6}; band SB32 sliding decode 17 <= B <= 32 = {master FA3 chain, D1, A1}; band SV1 sliding verify B=1 = {master chain, D3 pre + FA3, A1 pre + sk variant 5}. Print each candidate's median and 'PATH CHOSEN: <band>: <winner>'. Checks: A1 and A2 use attention master's acceptance rules (o rel-L2 vs an fp32 windowed reference with accurate sincosf rope; cache k within one E4M3 code by sign-magnitude ordinal on < 6% of bytes, v byte-exact, satfinite cvt accepted since sliding-cluster-c4-decode32-s2's probe showed torch saturates on this image), lens {1, 63, 2047, 2048, 2049, 80000}, ragged, kv_scale {1, 0.015625}, positions to 2^20, two alternating states with refilled inputs; D1-D3 as in C1. D2 runs exactly as in C1 (its own gates). Full layers, T > 32 and the GEMMs are master byte for byte. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, counter/ticket array, workspace and TMA map lives in the state, never in a module global (this includes the Triton sliding kernel's split counters: master's module-global _COUNTERS dict must NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per state and slice it). Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed load_inline module built at import like master's, with an in-process canary (synchronize + 'launching'/'ok' prints, flush=True) after each first launch; on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line keeps master for that cell. GATE PER SOURCE PER CLASS: each source path has its own prepare-time gate = (a) correctness check (bound stated per source below) on two alternating states with refilled inputs under captured-graph replay, then (b) a cold-L2 CUDA-graph selector (256 MB flush before each replay, >= 41 interleaved reps, empty-graph baseline subtracted) that enables the cell only when new/master <= 0.98 (i.e. master/new >= 1.02) and asserts the enabled flag equals the printed winner. Print exactly one 'PATH CHOSEN: <source> <class>: <winner> [new X us vs master Y us]' line per source per class. Every added prepare() phase runs AFTER master's own setup (after _dl3_finish), each with its own start deadline measured from prepare()'s t_prepare0 (skip the phase with a fallback line if the deadline has passed) and a print of its own wall seconds plus the cumulative prepare wall time; master's own internal deadlines and its _sk_prepare budget do not move. Print compile seconds per load_inline module. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_32*` faster: SB32: A1 expected winner (attention ranks it 6.3 us ahead of Triton); FA3 chain ~29.5 us of nodes -> pre+sk ~16 us; 239.2 -> ~222-226 us (vs C1's ~229)
- `*layer=sliding load=verify_1 *` faster: SV1: A1 variant 5 replaces FA3 9.9 + combine 3.9 + prepare_varlen + ~5 small nodes with ~13 us; plus D2; ~-10..-14 us on 180
- `*layer=sliding load=decode_1 *` faster: SB1: A2 cluster C16 10.8 us vs _sliding_fused_kernel 13.2 us; ~-2 us on 118
- `*layer=sliding load=decode_16*` faster: SB16: D1 or A1 (the race is the experiment); >= C1's ~-9 us
- `*layer=full load=verify_1 *` faster: D2 (unpool T=8 + R)
- `*layer=full load=decode_*` flat: full layers untouched except D2's R at T=16/32; control
- `*load=verify_16*` flat: untouched; control
- `*load=verify_32*` flat: untouched; control

## Risk


C1 in this file means the queued combination combo-measured-rows-slide-batched-preattn-t8-tail (same mint, 2026-10-25T161651Z): its sources S1/S2/S3, their gates and checks are reused verbatim here.

Not the top rows: on SB32 and SV1 the decode-measured winner (D1 / D3) is raced against a kernel that has never run inside decode (A1), and SB1's master is raced against A2. Expected headline about +2..+2.5% if A1 and A2 win (decode_1 ~1.2%, decode_32 ~4%, verify_1 ~5% of their loads). Prepare phases (after _dl3_finish): D2 ~4 s (deadline 120 s); A1 checks + alternation ~10 s and A2 canary + checks ~8 s (deadline 150 s); D1/D3 Triton JIT + checks ~15 s (deadline 170 s); the four band races ~12 s (deadline 200 s); total ~50 s. Import-time compile: +2 modules (A1 sk-window ~35 s, A2 cluster ~40 s) on top of master's 3 (~120 s), ~200 s of the 15-minute run; 933d took 238 s end to end, so ~320 s expected. If either new build fails, its candidates drop out of the races with a fallback line; the D-sources still run.

## Conjecture

Attention's per-kernel ranking does not transfer to the layer one-for-one, because inside decode the attention side is bounded by graph nodes and by what the qkv GEMM leaves in L2, not by the attention kernel alone. Evidence for the conjecture: in attention the Triton kernel is a 13.0 us standalone launch at B=16 and loses 6.3 us to sk-window at B=32, yet in decode the same Triton kernel won 4-5% of the WHOLE decode_16 row (~9-11 us, more than the FA3 kernel it replaced costs) because it also removes ~6 neighbouring nodes. Prediction: SB32 picks A1 by a margin smaller than attention's 6.3 us over Triton (both remove the same nodes; A1 adds one: pre + sk), SB16 picks D1 (A1's ticket kernel has too few keys per CTA at B=16), SB1 picks A2 only if the cluster launch overhead stays under ~1 us inside the graph, and SV1 picks A1 over D3. Second interaction: A1's _pre_attention_kernel writes K/V into the cache and the sk kernel reads the same pages within microseconds; in attention the qkv input was cold, here it is hot from the qkv GEMM; a larger A1 margin in decode than in attention would show the L2 effect. Confirm/refute: the four race lines (every candidate's us), the sliding decode_16/32, verify_1 and decode_1 rows vs 933d and vs C1 (same-day run), and the profile kernel rows (splitkv_attn_sk_kernel + _pre_attention_kernel at Sd32 should sum to ~16 us; sliding_cluster_kernel at Sd1 ~10.8 us).
