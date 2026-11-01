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
review: proceed
status: hold
review_at: 2026-10-25T162310Z
review_companions: null
stage: null
claimed_at: null
base_sha: null
base_blob: null
branch: null
worktree: null
mode: null
restack_of: null
stale_base: null
hold_reason: "superseded: remainder equals sliding-sk-window-port-decode32-verify1 + sliding-cluster-c16-decode1-port, both queued (decode minter, 2026-10-25)"
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

## Adversarial review

Reviewed 2026-10-25T162310Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a scored combination run. The cache layout and FP8 scale conventions match decode's exactly, and the run time stays well inside 15 minutes. Five build notes are binding: the attention module's ticket arrays move into state; A1's variant and grid are raced or ported, not hard-coded; the race times the qkv GEMM through the o-GEMM input; every A1/A2 gate keeps page_size == 1; and the text must say clearly that this is not all of C1 (S3 c1/c3/c4 are missing), so the headline is not compared with C1 on the full-layer rows.

### Justification
A. The mechanism is plausible and the run can test the conjecture, because every band prints every candidate's time in the same interleaved replay set and adds the 'PATH CHOSEN' line. In 933d, the sliding batched rows still run the FA3 chain: FA3 13.6/18.2 + combine 4.3/2.9 + prepare_varlen 2.3/2.2 + FusedQKNorm, casts, store_cache and FusedRMSNormGate. That is about 28-30 us of nodes. D1 removed about 9-11 us of it before. Attention's log 252abb7f shows pre + sk variant 6, grid p4, at 19.5 us net against Triton's 25.8 us at B=32. An in-layer margin for A1 over D1 at SB32 is credible, but it will be smaller than 6.3 us, as the conjecture itself predicts. SB1: attention's cluster C=16 kernel row is 10.8 us against _sliding_fused_kernel's 13.2 us in 933d, so roughly -2 us on a 118 us row. That is about 1.7%, close to the selector's 1.02 threshold and to the run's ±1.5% interval, so whatever SB1 decides will not show clearly in the headline. Its race line is the evidence. The expected directions match TASK weights: sliding rows are weighted 46, so sliding decode_32 (the worst ratio, 1.080x) and verify_1 are the leverage points. The headline estimate of +2..2.5% is optimistic. A1 has never run inside decode, and the C1 parts dropped here (below) give up Fd16/Fd32/Fv1 gains that C1 expects. WEAKEST LINK: A1's configuration is pinned from attention's standalone choices and is partly wrong or unspecified. Attention never gated sk-window at B=2..16 (its log says 'master (Triton), not gated in this restack'). No grid spec is given for variant 6 at SB16. For sliding verify B=1, attention's own runs chose single-level variant 4 at grid p8 (logs bd0ddd41, 94523c33), not variant 5 as the hypothesis states. If A1 is entered with the wrong variant or grid, it loses the race for configuration reasons, and the conjecture ('the ranking does not transfer') is then confirmed spuriously. Fix: race A1's own {variant 4/5 (SV1) or 6/7 (decode)} x {p2, p4, p8, sm} with the ported _sd/_ps gate-band code, and enter only its best into the three-way race. Or port attention master's _SD_ROUTE/_PS_ROUTE choices exactly as its log printed them: B=17..32 is variant 6, grid p4. KV/FP8 match (EXTRA_NOTE), verified in code. Both problems use float8_e4m3fn [pages, page_size, 2, 64] contiguous, flat row = cache_loc = page*page_size + offset, row = 2 kv heads x 64 B. Attention's _pre_attention_kernel and the cluster kernel write k at (loc*HKV + kh)*D. Both store k/k_scale and v/v_scale, and both apply score scale sm_scale*qk_scale_factor*log2e*k_scale with output times v_scale, which is decode's convention (key = e4m3 * k_scale). The qkv row layout is identical: q 16x64 | k 2x64 | v 2x64 | gate 16x64, gate at element 1280 = decode's SK_QKV_GATE_OFF = attention's (HQ+2HKV)*D. QK-norm is scaleless in both, the output RMSNorm is scaleless then times sigmoid(gate) in both, and both use interleaved RoPE with theta 500000. Known byte differences: (1) attention normalises k in fp32 and encodes it straight to E4M3, while master rounds through bf16 (FusedQKNorm) first; (2) Triton tl.sin/tl.cos are approximate (LESSONS line 140). Both give a few k bytes off by one E4M3 code, which is what the hypothesis's '< 6% within one code' bound covers and which is far inside the judge's 8% cache bound. v is byte-exact at kv_scale 1 and 1/64 (power-of-two division). The cluster kernel rounds q*qk_scale_factor to E4M3 rather than rounding q and then scaling. That is the same E4M3 precision, so the 'no coarser than E4M3' rule holds, and attention's judge accepted it. BUDGET (EXTRA_NOTE): 933d ran 237.5 s end to end (okc created->finished). Master builds 3 modules at import; sk/splitkv builds in the logs take 30-50 s each. Adding A1, the SK_PSEUDO TU holding all 8 variant instantiations (likely 45-60 s rather than 35), and A2 (about 40 s, plus about 40 s more if its f16 mma fallback build triggers) adds about 85-140 s. With prepare's +50 s, the worst case is about 375-430 s, against 900 s. That is safe with more than 450 s of margin. Prepare's 300 s limit is also safe: master's prepare took 4.5 s, the last phase's start deadline is 200 s, and the race takes about 12 s.

### Confounders on the current kernel
- C1 itself (combo-measured-rows-slide-batched-preattn-t8-tail): this is NOT 'C1 plus attention ports'. It carries C1's S1 (as D1) and S2 (as D2), but S3 only on sliding verify B=1 (D3). C1's S3 c1 (full decode 1<B<=64, +4.7/+3.3% own-base), c2 at T=128/256 (sliding verify_16/32), c3 (full verify B>1) and c4 (full verify_1) are all missing. On Fd16, Fd32, Fv1, Sv16 and Sv32 the two combos therefore differ by S3, not by the race, so a same-day headline comparison with C1 cannot be attributed. The rows that compare cleanly with C1 are Sd1, Sd16, Sd32 and Sv1 only, and then only if C1's S3 c2 selector at T=8 decides the same way as this run's D3 candidate.
- small-t-launch-minimal-dense-chain-pdl / dense-chain-s3 (PDL chain at T<=32): A1's pre + sk and A2's cluster kernel are ported without griddepcontrol. Inserting them between a PDL-launched qkv GEMM and o GEMM can cost the PDL overlap that master's or D1's path keeps. A race that times only the attention kernels would hide this; a race over qkv -> o-GEMM input exposes it. The conjecture's L2-hot claim is likewise only testable if the race graph includes the qkv GEMM with a cold-L2 flush before it.
- sliding-decode1-fused-restack-on-pdl (in stack-align-par-sidestream-sliding1): this is SB1's incumbent, _sliding_fused_kernel 13.2 us. A2 is raced directly against it, so attribution is clean. But the incumbent fuses RoPE and the cache store and may sit inside the PDL chain, so the head-to-head must run on the same captured neighbours.
- full-decode-sk-ticket-restack-b1-b64 / dl3-piece-a: decode's _SK module stays byte-identical and the new module builds the same TU source under a new name. That is harmless only if the tickets are separate per-state arrays (see rules_flags). If the new module reuses attention's module-global _ps_ticket/_ps_gticket, it is a cross-state shared counter. That works while replays are serialized, but it breaks the stated two-state invariant.

### Interference and A/B needs
No companion is warranted. The run is self-attributing on its four bands: every candidate's time and the winner are printed per band from one interleaved replay set, and the shape rows can be compared with 933d. C1 is already queued from the same mint, so a same-day C1 result serves as the drift reference on Sd1, Sd16, Sd32 and Sv1 at no extra scored cost. An ab-revert of any landed hypothesis would not make either a gain or a loss less ambiguous, because the race already pits master's incumbent against each source inside the same layer. The one ambiguity that matters, A1 losing because its variant or grid was wrong, is fixed by the in-run A1 variant x grid sub-race required in justification, not by another scored run. Optional, no extra run: if the builder adds C1's S3 c1-c4 verbatim (Triton only, no new module), the whole run differs from C1 only by the race, and a full-headline A/B against C1 becomes valid.

### Open questions
- Which A1 variant and grid does SV1 race: variant 5 as written, or attention's measured variant 4 at grid p8? Recommend a sub-race over {4, 5} x {p4, p8, sm}, entering the winner.
- Which graph is timed in each band race? Require the qkv GEMM through the o-GEMM input, cold L2 before the qkv GEMM, with PDL neighbours as master captures them. Otherwise the conjecture about L2 reuse and nodes is untestable.
- Should S3 c1-c4 be included so the run is literally C1 + race (clean A/B against same-day C1 on every row)? If not, the summary should drop the 'contains C1' framing and compare with C1 only on Sd1, Sd16, Sd32 and Sv1.
- prepare() is called per state copy: can the race choose different winners for the two states? Print the decisions of both calls. If they differ, the profile rows mix paths, so report both rather than reading a single row.
- SB1's expected gain of about 1.7% of the row is at the 1.02 threshold. Report the race margin even when master is kept, so a near-miss is not read as a refutation.


## Implementation notes

NO EDIT MADE (worktree submission.py untouched, base 49aaf9da). Reconciliation against current master:
- D1 (sliding batched Triton S=16), D2 (R variant 0 / unpool NF=1) and D3 (fused pre-attention FA3 feed) are all already on master via ec89382b (combo-measured-rows-slide-batched-preattn-t8-tail, 1.3142x), which also carries C1's S3 c1-c4 that this md says are missing. Master has the per-source/per-class gate and 'PATH CHOSEN: <source> <class>' machinery (submission.py ~L5973-6908).
- What remains is only A1 (new muse_decode_sk_window_v1 CUDA module, pre-attention kernel, variant table, PSEUDO prefix loader, variants 4/5/6 x grids) and A2 (muse_decode_slide_c16_v1 cluster-C16 module) plus a per-band race over them. Neither exists in decode master (no PSEUDO / sk_window / slide_c16 symbols).
- Not implemented: a faithful port is a multi-thousand-line CUDA translation unit (muse_attention_decode/submission.py sk TU ~L1600-3350, cluster TU ~L3600-4900) with per-state tickets, two new load_inline builds, variant x grid sub-race and a qkv-through-o-GEMM race graph (adversarial review's binding notes). It cannot be compiled or run on this node and goes out as an unchecked scored submit; a blind port risks an illegal-address/hang that no prepare-time fallback can catch. Orchestrator should hold; if wanted, run it as an okc check first or split A1 / A2 into separate smaller ports.

## Rules check
- [x] n/a: no diff produced (git diff empty), nothing to violate; lint not run since file unchanged.
