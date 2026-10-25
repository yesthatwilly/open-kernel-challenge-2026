---
problem: muse_attention_decode
slug: combo-tiled-preattn-neighbour-regate-c320
created: 2026-10-25T161609Z
base: master
area: "combination (conjectural: tiled pre-attention as every pair's first node, sd B=16 and skF selectors re-decided, + chunk320)"
change: >
  COMBINATION 3 of 3, a conjectural neighbour pairing: the pre-attention kernel as the shared
  first node of every two-node pair, with the downstream selectors re-deciding once it changes.
  Built on current master 7e12891 (blob 5cf8fae). Carries SOURCE A (chunk320 at full decode_1,
  exactly as in combination 1) so the run has a measured headline lift, and differs from
  combination 1 only in where the tiled pre-attention goes and which selectors are re-run; rows
  that differ between the two runs are the readout. SOURCE A = full-decode1-cuda-splitkv-chunk320
  (branch hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job
  8d4771feca9b4e18: full decode_1 19.49 us = splitkv_attn_kernel<320> 11.0 +
  splitkv_combine_kernel<320,16> 4.5 + _pre_attention_kernel 1.3; master's two-level stream-K is
  22.40 us in 252abb7f and 22.53 in 917a9689, so 0.87x of master on the same row, paired vs SGLang
  1.604 vs 1.393/1.395). It was never compared with the two-level kernel: it harvested neutral on
  an older base and s3 landed after it. Master's _SPLITKV_CUDA is byte-identical to that run's
  base, so the CUDA ports cleanly. Port: the templated splitkv_attn_kernel<CHUNK> and
  splitkv_combine_kernel<CHUNK, NW> with launch_attn<320> / launch_combine<320,16> (512-thread
  combine), the strided issue_meta loop, the __host__ __device__ constexpr smem helpers (audit
  fix), instantiated ONLY at <320> and <320,16>, as a NEW load_inline module muse_full_decode_c320
  (TORCH_CUDA_ARCH_LIST 9.0a, -O3 -lineinfo, -std=c++20 in extra_cuda_cflags, binaryVersion == 90
  check, print regs/local/smem). Master's v1 module and its <1024> code stay untouched. Python:
  _full_decode_c320(data, q, qkv) = master _pre_attention_kernel (unpaired, exactly as master's
  batch-1 branch) then c320 A + combine, max_chunks = ceil(page_table.shape[1] / 320), grid 2 x
  SM. Gate class FD1 = config window == 0, q_len 1, batch 1, tp8, d 64, page 1 (the same predicate
  as master's _shape_gate at B=1). GATE A (FD1): (i) correctness: the chunk320 run's batch-1 cases
  (lens [80000] and [5000], kv 1 and 0.25, grid device and 3) plus 1 / 63 / 64 / 4097 keys, rel L2
  < 2e-2 vs master's FP32 _splitkv_reference AND < 5e-3 vs master's two-level output on the same
  inputs, finite, cache rows equal to master's pre-attention rows (same kernel); (ii) two-state
  alternating capture-once replay (A B B A A B, refilled inputs, NaN-poisoned outputs) at 80000
  and 4097 keys; (iii) speed: interleaved cold-L2 CUDA-graph replay (256 MB flush, empty graph
  subtracted, 41 reps, median) of [pre-attn + c320 A + combine] vs master's [pre-attn + two-level
  <4,true,true> grid SM] at B=1 x 80000; route only if c320 / master <= 0.98 (expected ~0.87).
  Print 'PATH CHOSEN: full decode_1 [A chunk320]: <route|master two-level> (c320 X us vs two-level
  Y us, ratio R)'. SOURCE B = preattention-tiled-gate-large-tokens (branch
  hyp/muse_attention_decode/preattention-tiled-gate-large-tokens, impl 880b01b, job
  c2ebf6ed05824692): _pre_attention_tiled_kernel verbatim, as in combination 1, but offered to
  every class whose step starts with a separate pre-attention node: sliding verify SV1/SV16/SV32
  and full verify FV1/FV16/FV32 (q_len 8), full decode FD16 (B 2..16) and FD32 (B 17..64) on
  master's [pre-attn + v2] path, full decode FD1 inside SOURCE A's [pre-attn + c320] pair, and
  sliding decode SD32 (B 17..32) inside master's [pre-attn + sk window] pair (_sd_run's
  _ps_prepare_q call). GATE B per class exactly as in combination 1 (byte/ulp exactness vs
  master's kernel, whole-step interleaved cold-L2 replay, route at <= 0.98). For the pair classes
  the 'whole step' is the pair kernel_fn would run, so the downstream kernel is held fixed and
  only the first node changes. SOURCE E = sliding-decode-sk-window-route (branch
  hyp/muse_attention_decode/sliding-decode-sk-window-route, impl 376be02, job 118c6de9704146f3):
  its B=2..16 band of the sk-window gate. There the best pair (<4 stages, ticket, PSEUDO TOK=1>
  grid p8 = 128 CTAs) was 16.2 us net vs Triton 16.3 (1.006x) with 'pre-attention alone 3.7 us
  net', so the pre-attention node is ~23% of that pair. Master's _sd_gate_band is already generic
  over batch: add band 2 (B=16 x 80000, incumbent master Triton _sliding_fused_kernel, which is
  what kernel_fn runs there) to _sd_setup_body's band loop AFTER GATE B has frozen the
  pre-attention choice, with _sd_run using the tiled kernel when GATE B routed SD16 (add SD16 = B
  2..16 sliding decode to GATE B's class list, timed as [pre + sk p8] vs [tiled + sk p8]); route
  band 2 only at pass-2 Triton / pair >= 1.03x (time ratio <= 0.97, inside the <= 0.98 rule), with
  master's replay check (two states, refilled inputs, tickets back to 0) on B=16 ragged lens. Band
  3 (B=17..32) keeps master's decision, but is re-timed with the tiled pre-attention if GATE B
  routed SD32, and master's own 1.03x-vs-cluster rule decides. Then re-run master's _skf_gate_band
  at B=16 and B=32 with the separate pre-attention set to whatever GATE B chose for FD16/FD32
  (FUSED_PRE stays off unless >= its existing 1.02x; master measured 0.988-0.996x / 1.014-1.015x).
  Print PATH CHOSEN per class for SOURCE B, per band for SOURCE E, and the skF re-decision. PHASES
  AFTER MASTER: c320 build ~40 s (deadline 120 s); GATE A ~10 s; GATE B over 10 classes ~25 s
  incl. Triton JIT (deadline 60 s); band-2 sd gate ~12 s and replay ~5 s (deadline 40 s); skF
  re-gate ~8 s (deadline 30 s). ~100 s added. Any phase past its deadline leaves its classes on
  master. IMPORT BUDGET (no prepare() on this problem: everything runs at import and counts toward
  the 15-min run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s
  each). Master's import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1,
  _ps_setup, _skf_setup, diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its
  own thresholds. Every added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock
  and is SKIPPED (master kept for its classes, PATH CHOSEN says so) if the clock is past 420 s
  when it would start or past its own deadline when it finishes. RULES (every path): CUDA-graph
  capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config
  only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable
  Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16
  out; every key of the context attended; no child processes, no timing in kernel_fn. The judge
  prepares TWO states and rotates them: no module/static global may hold a device pointer into a
  state's tensors, a TMA map or a state-sized buffer (the c320 path has no tickets; its workspaces
  are torch.empty per call, sized from page_table.shape[1] and batch). Route tables are frozen
  (MappingProxyType / module bools set once at import) before any capture.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "SOURCE A as in combination 1: 22.4 -> ~19.5 us (~+0.85% headline); the tiled first node may shave a further 0.1-0.2 us"
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: tiled pre-attention 4.2 -> 2.7 us measured twice (~+0.14% headline)
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: "conjecture: with the tiled first node the sk-window pair at B=16 (16.2 vs Triton 16.3 us net) clears 1.03x; 15.5 -> ~15.0 us, ~+0.15% headline. Flat if band 2 is not routed"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: pre-attention is 1.9 us of the 17.7 us sk-window pair (4.0 us net eager in the gate); -0.3 to -0.8 us if GATE B routes SD32
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: "pre-attention 2.2/2.4 us of 111/228 us; any gain is below this row's run-to-run drift (~6% at decode_32); read the kernel row"
  - shapes: "*layer=full load=verify_* *"
    direction: flat
    why: paired layout gained 0.1 us in the source; expected gated off
  - shapes: "*layer=sliding load=decode_1 * *layer=sliding load=verify_1 *"
    direction: flat
    why: cluster C=16 has no pre-attention node; SV1 8 tokens expected gated off
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T162052Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T162052Z
base_sha: 1e8d6d6adbaadd4ecaf72ebd01c75f24004e81bc
base_blob: 5cf8fae31e5b84c36efeeb4569763eb9b06e1c8a
branch: hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320
worktree: .worktrees/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320
mode: null
restack_of: null
stale_base: null
---

# combo-tiled-preattn-neighbour-regate-c320

## Change

COMBINATION 3 of 3, a conjectural neighbour pairing: the pre-attention kernel as the shared first node of every two-node pair, with the downstream selectors re-deciding once it changes. Built on current master 7e12891 (blob 5cf8fae). Carries SOURCE A (chunk320 at full decode_1, exactly as in combination 1) so the run has a measured headline lift, and differs from combination 1 only in where the tiled pre-attention goes and which selectors are re-run; rows that differ between the two runs are the readout. SOURCE A = full-decode1-cuda-splitkv-chunk320 (branch hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job 8d4771feca9b4e18: full decode_1 19.49 us = splitkv_attn_kernel<320> 11.0 + splitkv_combine_kernel<320,16> 4.5 + _pre_attention_kernel 1.3; master's two-level stream-K is 22.40 us in 252abb7f and 22.53 in 917a9689, so 0.87x of master on the same row, paired vs SGLang 1.604 vs 1.393/1.395). It was never compared with the two-level kernel: it harvested neutral on an older base and s3 landed after it. Master's _SPLITKV_CUDA is byte-identical to that run's base, so the CUDA ports cleanly. Port: the templated splitkv_attn_kernel<CHUNK> and splitkv_combine_kernel<CHUNK, NW> with launch_attn<320> / launch_combine<320,16> (512-thread combine), the strided issue_meta loop, the __host__ __device__ constexpr smem helpers (audit fix), instantiated ONLY at <320> and <320,16>, as a NEW load_inline module muse_full_decode_c320 (TORCH_CUDA_ARCH_LIST 9.0a, -O3 -lineinfo, -std=c++20 in extra_cuda_cflags, binaryVersion == 90 check, print regs/local/smem). Master's v1 module and its <1024> code stay untouched. Python: _full_decode_c320(data, q, qkv) = master _pre_attention_kernel (unpaired, exactly as master's batch-1 branch) then c320 A + combine, max_chunks = ceil(page_table.shape[1] / 320), grid 2 x SM. Gate class FD1 = config window == 0, q_len 1, batch 1, tp8, d 64, page 1 (the same predicate as master's _shape_gate at B=1). GATE A (FD1): (i) correctness: the chunk320 run's batch-1 cases (lens [80000] and [5000], kv 1 and 0.25, grid device and 3) plus 1 / 63 / 64 / 4097 keys, rel L2 < 2e-2 vs master's FP32 _splitkv_reference AND < 5e-3 vs master's two-level output on the same inputs, finite, cache rows equal to master's pre-attention rows (same kernel); (ii) two-state alternating capture-once replay (A B B A A B, refilled inputs, NaN-poisoned outputs) at 80000 and 4097 keys; (iii) speed: interleaved cold-L2 CUDA-graph replay (256 MB flush, empty graph subtracted, 41 reps, median) of [pre-attn + c320 A + combine] vs master's [pre-attn + two-level <4,true,true> grid SM] at B=1 x 80000; route only if c320 / master <= 0.98 (expected ~0.87). Print 'PATH CHOSEN: full decode_1 [A chunk320]: <route|master two-level> (c320 X us vs two-level Y us, ratio R)'. SOURCE B = preattention-tiled-gate-large-tokens (branch hyp/muse_attention_decode/preattention-tiled-gate-large-tokens, impl 880b01b, job c2ebf6ed05824692): _pre_attention_tiled_kernel verbatim, as in combination 1, but offered to every class whose step starts with a separate pre-attention node: sliding verify SV1/SV16/SV32 and full verify FV1/FV16/FV32 (q_len 8), full decode FD16 (B 2..16) and FD32 (B 17..64) on master's [pre-attn + v2] path, full decode FD1 inside SOURCE A's [pre-attn + c320] pair, and sliding decode SD32 (B 17..32) inside master's [pre-attn + sk window] pair (_sd_run's _ps_prepare_q call). GATE B per class exactly as in combination 1 (byte/ulp exactness vs master's kernel, whole-step interleaved cold-L2 replay, route at <= 0.98). For the pair classes the 'whole step' is the pair kernel_fn would run, so the downstream kernel is held fixed and only the first node changes. SOURCE E = sliding-decode-sk-window-route (branch hyp/muse_attention_decode/sliding-decode-sk-window-route, impl 376be02, job 118c6de9704146f3): its B=2..16 band of the sk-window gate. There the best pair (<4 stages, ticket, PSEUDO TOK=1> grid p8 = 128 CTAs) was 16.2 us net vs Triton 16.3 (1.006x) with 'pre-attention alone 3.7 us net', so the pre-attention node is ~23% of that pair. Master's _sd_gate_band is already generic over batch: add band 2 (B=16 x 80000, incumbent master Triton _sliding_fused_kernel, which is what kernel_fn runs there) to _sd_setup_body's band loop AFTER GATE B has frozen the pre-attention choice, with _sd_run using the tiled kernel when GATE B routed SD16 (add SD16 = B 2..16 sliding decode to GATE B's class list, timed as [pre + sk p8] vs [tiled + sk p8]); route band 2 only at pass-2 Triton / pair >= 1.03x (time ratio <= 0.97, inside the <= 0.98 rule), with master's replay check (two states, refilled inputs, tickets back to 0) on B=16 ragged lens. Band 3 (B=17..32) keeps master's decision, but is re-timed with the tiled pre-attention if GATE B routed SD32, and master's own 1.03x-vs-cluster rule decides. Then re-run master's _skf_gate_band at B=16 and B=32 with the separate pre-attention set to whatever GATE B chose for FD16/FD32 (FUSED_PRE stays off unless >= its existing 1.02x; master measured 0.988-0.996x / 1.014-1.015x). Print PATH CHOSEN per class for SOURCE B, per band for SOURCE E, and the skF re-decision. PHASES AFTER MASTER: c320 build ~40 s (deadline 120 s); GATE A ~10 s; GATE B over 10 classes ~25 s incl. Triton JIT (deadline 60 s); band-2 sd gate ~12 s and replay ~5 s (deadline 40 s); skF re-gate ~8 s (deadline 30 s). ~100 s added. Any phase past its deadline leaves its classes on master. IMPORT BUDGET (no prepare() on this problem: everything runs at import and counts toward the 15-min run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s each). Master's import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1, _ps_setup, _skf_setup, diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its own thresholds. Every added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock and is SKIPPED (master kept for its classes, PATH CHOSEN says so) if the clock is past 420 s when it would start or past its own deadline when it finishes. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key of the context attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (the c320 path has no tickets; its workspaces are torch.empty per call, sized from page_table.shape[1] and batch). Route tables are frozen (MappingProxyType / module bools set once at import) before any capture.

## Expected

- `*layer=full load=decode_1 *` faster: SOURCE A as in combination 1: 22.4 -> ~19.5 us (~+0.85% headline); the tiled first node may shave a further 0.1-0.2 us
- `*layer=sliding load=verify_32 *` faster: tiled pre-attention 4.2 -> 2.7 us measured twice (~+0.14% headline)
- `*layer=sliding load=decode_16 *` faster: conjecture: with the tiled first node the sk-window pair at B=16 (16.2 vs Triton 16.3 us net) clears 1.03x; 15.5 -> ~15.0 us, ~+0.15% headline. Flat if band 2 is not routed
- `*layer=sliding load=decode_32 *` faster: pre-attention is 1.9 us of the 17.7 us sk-window pair (4.0 us net eager in the gate); -0.3 to -0.8 us if GATE B routes SD32
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: pre-attention 2.2/2.4 us of 111/228 us; any gain is below this row's run-to-run drift (~6% at decode_32); read the kernel row
- `*layer=full load=verify_* *` flat: paired layout gained 0.1 us in the source; expected gated off
- `*layer=sliding load=decode_1 * *layer=sliding load=verify_1 *` flat: cluster C=16 has no pre-attention node; SV1 8 tokens expected gated off

## Risk

Medium risk: the same two judged sources as combination 1 plus a gate band that ran clean on the judge (sliding-decode-sk-window-route printed the B=16 numbers and declined at its 1.05x rule). The new risk is selector coupling: the sd and skF gates must time the pre-attention kernel kernel_fn will really launch, so GATE B must be frozen first and _sd_run / the skF [pre-attn + v2] timing must read that frozen choice.

## Conjecture

One source's selector re-decides once its neighbour changes. The pre-attention kernel is the first node of every two-node pair master routes (sk window at B=17..32, v2 at full decode, c320 at B=1, every verify), and two of master's selectors sit just below their thresholds because of it: the sk-window pair at B=16 was 1.006x vs Triton with a 3.7 us-net pre-attention node, and FUSED_PRE (which folds pre-attention in) was 1.014-1.015x at B=32 against the separate pair. Claim: making the separate first node cheaper (tiled, 64 programs of 4 warps instead of 18 x tokens 1-warp programs) moves those selectors in opposite directions: it lifts the sk pair at B=16 over its gate (a new route on a 46-weight row) and pushes FUSED_PRE further below its gate (no route change, but a measurable margin shift). Untested premise: the tiled kernel's win was measured only at >= 64 tokens (and +0.1 us at 8); whether it still wins at 16-32 tokens is exactly what GATE B's SD16/SD32/FD16/FD32 pre-attention lines will show. Confirm: GATE B routes SD16 and SD32, the band-2 '[sd-gate] B=16 ... pass 2 ... Triton / pair' rises from ~1.006x to >= 1.03x and routes, the skF B=32 ratio drops below ~1.01x, and the judged sliding decode_16 profile shows _pre_attention_tiled_kernel + splitkv_attn_sk_kernel instead of _sliding_fused_kernel at <= 15.2 us. Refute: GATE B declines at 16-32 tokens (tiled not faster with so few programs) or the B=16 pair stays below 1.03x; compare the sliding decode_16/32 rows against combination 1's run, which keeps master there.

## Adversarial review

Reviewed 2026-10-25T162052Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, with four required edits. The run's measurable part is SOURCE A (c320 at full decode_1, about +0.85% headline, same as combination 1). The neighbour conjecture can only be read from in-process gate prints, so pass 2 of band 2 must time the master-pre pair and the tiled-pre pair head to head, and band 2 must keep master's 1.05x-vs-Triton rule. The skF re-gate is noise by construction: drop it, or keep it as print-only and remove it from the confirm/refute criteria.

### Justification
SOURCE A is well supported: job 8d4771fe measured 19.49 us against master's two-level 22.40/22.53 us on the same row, and GATE A (cold-L2 interleaved replay, route at <= 0.98) protects it. SOURCE B is supported only where it was measured. In c2ebf6ed the tiled kernel ran 2.1 us at sliding verify_16 (128 tokens) and 2.8 us at verify_32 (256 tokens) against master's 2.8/4.2 us. It lost or was flat at 8 tokens and at sliding decode_1. Master's untiled kernel is already only 1.9-2.4 us at 16-32 tokens (judged profile: full decode_16 2.2, sliding decode_32 1.9, full decode_32 2.4). The tiled kernel cannot go much below about 1.3-1.5 us because of launch and latency cost, so the realistic saving at 16-32 tokens is 0-0.5 us. WEAKEST LINK: the claim that band 2 (sliding decode B=16) will route. The 1.006x it starts from is source E's pass-1 first-pass MINIMUM (16.2 vs 16.3 us net, a biased number per master's own comment), not a pass-2 head-to-head. To reach master's real rule for a Triton incumbent (1.05x, `need = 1.05` in _sd_gate_band), the pair must fall to about 15.5 us net, a saving of about 0.7 us. That is more than the whole plausible tiled saving at 16 tokens. Even the hypothesis's own 1.03x needs about 0.4 us. The predicted outcome is therefore 'band 2 declines', which is a fine refutation, but the expected 'faster' on sliding decode_16 is optimistic. The hypothesis also quietly changes band 2's threshold from master's 1.05x (Triton incumbent) to 1.03x. A route at 1.03-1.05 would then come from the threshold, not from the neighbour. Classes that cannot route by construction: GATE B uses a whole-step ratio <= 0.98. FD16/FD32 whole steps are 111/228 us with a 2.2/2.4 us pre node, so even a zero-cost pre node gives at best 0.980/0.989. FV16/FV32 (163/302 us, pre 2.8/4.2) are the same. Those four classes can never route, so the skF re-gate at B=16/32 always runs on master's unchanged separate pre-attention and only re-times identical code. Even if FD32 did route, the shift would be about 0.4 us on a 228 us denominator, about 0.002 in the ratio. That is below the gate's own repeatability (B=16 has read 0.988-0.996 across runs) and far below the about 6% judged-row drift, so it is not an interaction test. SD32 is the one pair class with a real chance: about 0.39 us is needed on a 19.5 us gate-net pair whose pre node is 4.0 us net in the gate. Headline readout: c320 adds about +0.85% and everything else is at most about 0.3%, all inside the ±1.5% CI. The run therefore cannot be told apart from combination 1 on headline, and judged decode_16/32 rows (0.3-0.5 us of 15.5-17.7 us, 2-3%) sit inside the ~6% row drift. The only interpretable readouts are (1) whether c320 routes and the full decode_1 row moves 22.4 -> ~19.5 us, which is outside drift, (2) the GATE B per-class pre-node lines, (3) band 2's pass-2 ratio, and (4) which kernel names appear in the judged decode_16/32 profiles. Import budget: master's 252abb7f ran 470 s from start to finish, tests and timing included, so import is somewhat less than that and the +100 s fits the 15 minutes. The 420 s skip clock is a real risk: the band-2 phase, which carries the conjecture, comes late and could be skipped. Every phase should print its elapsed IMPORT_T0 time so a skip is visible.

### Confounders on the current kernel
- restack-sliding-decode-sk-window-on-cluster-master (master 7e12891): owns _sd_gate_band and its threshold rule (1.03 vs cluster, 1.05 vs Triton). Band 2 at 1.03 vs Triton is a threshold change, so a band-2 route between 1.03x and 1.05x would come from the threshold, not the tiled neighbour, unless the master-pre pair is timed in the same pass.
- sliding-decode-sk-window-route (source E, unlanded B=16 numbers): the 1.006x baseline is a pass-1 biased minimum from another build. Comparing a new pass-2 number with it mixes measurement method with the change. The comparison must be made in-pass.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (FUSED_PRE / skF): the skF re-gate can only shift if GATE B routes FD16/FD32, which the 0.98 whole-step rule makes impossible (pre node is 2.0%/1.1% of the step). The re-gate is a re-measurement of master and its 'margin shift' readout is noise.
- full-decode1-streamk-s3-two-level-ticket: the incumbent c320 must beat at full decode_1. GATE A times against the real <4,true,true> grid-SM pair, so it is not confounded, provided the comparison graph includes master's pre-attention, as stated.
- paired-heads-launch-cleanup / -ab-tune-batch-gate: the pre-attention kernel writes the paired q layout on full verify at B>1. The tiled kernel must reproduce that layout byte-exactly for FV16/FV32, but those classes cannot route anyway (see above). Byte-exactness against master's paired kernel is still required for any class that does route.
- fused-preattention-triton (the _pre_attention_kernel itself): source B's tiled kernel replaces it. Exactness against it (byte/ulp, cache rows) is GATE B's correctness gate, so no hidden numerical confound.

### Interference and A/B needs
No scored companion is warranted. The natural control already exists: combination 1 (combo-naive-c320-fd1-tiled-preattn-verify, queued) carries the identical SOURCE A and keeps master on the pair classes. Rows that differ between the two runs are the designed readout. An extra scored run would only add another sample inside the ~6% row drift. The interaction test does need to be self-contained in THIS run's import, at no extra cost: (1) In band 2's pass 2, time FOUR graphs interleaved in the same set: Triton, [master pre + sk p8], [tiled pre + sk p8], and the two pre nodes alone. Print Triton / pair for BOTH pairs. The difference between the two ratios IS the neighbour effect, measured in one pass with no drift. Do the same for band 3 (SD32) against cluster C=4. (2) Keep master's 1.05x rule against a Triton incumbent for band 2. If 1.03 is kept, the master-pre pair line from (1) is mandatory and the route reason must say whether the master-pre pair also cleared it. (3) Drop the skF re-gate. FD16/FD32/FV16/FV32 cannot route under GATE B's whole-step <= 0.98 rule (best case 0.980/0.989 even at zero pre cost), so the re-gate re-times master. Optionally keep the FD/FV GATE B lines as print-only diagnostics of the pre node at 16/32/128/256 tokens, which do answer the 'untested premise'. (4) Run band 2 BEFORE any optional diagnostics, and print elapsed IMPORT_T0 at every phase boundary, so a 420 s skip is visible and cannot be mistaken for a refutation.

### Open questions
- How long does master's import take on the judge? 252abb7f is 470 s for the whole run including tests and timing. If import is already ~380 s, the 420 s clock will skip band 2 and the conjecture goes untested while still spending a scored submit. Consider building c320 in parallel or early (it is independent of master's gates) only if that does not change master's measured selectors.
- Was source E's band-2 1.006x ever a pass-2 number? The log shows only pass-1 lines for B=16 (grid 128 p8 at 16.2 us net). If pass 2 at B=16 has never been measured, the 'starting point' of the conjecture is unknown.
- Is the tiled kernel's program count at 16-32 tokens (64 programs of 4 warps, per the conjecture) actually fewer and faster than 288-576 one-warp programs, or does it under-fill 132 SMs at that size? The GATE B SD16/SD32 pre-node lines answer this and should be printed even when the class declines.

