---
problem: muse_attention_decode
slug: combo-b1-chain-c320-fused-pre-pdl
created: 2026-10-25T161608Z
base: master
area: "combination (conjectural: full decode_1 chunk320 + fused pre-attention prologue + PDL to combine)"
change: >
  COMBINATION 2 of 3, a conjectural B=1 launch-chain pairing (not the top rows). Built on current
  master 7e12891 (blob 5cf8fae). Full decode_1 is the highest-leverage full row (weight 16 in the
  smallest load, 905 us: ~0.29% headline per us). On it, each measured design leaves 1.0-1.4 us
  per graph node of non-kernel time: master two-level 20.4 us of kernels in a 22.4 us shape (2
  nodes), chunk320 16.8 us in 19.49 (3 nodes), fused two-level 20.3 in 21.6 (1 node). This
  combination puts the fastest kernel design (chunk320) on the fewest effective nodes by stitching
  two sources that each removed one node boundary elsewhere. SOURCE A =
  full-decode1-cuda-splitkv-chunk320 (branch
  hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job
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
  Y us, ratio R)'. SOURCE C = full-decode1-sk-two-level-fused-preattn (branch
  hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn, impl 2236339, job
  2038a530edd04309; its gate 20.74 vs 21.98 us = 1.060x, judged 21.60 vs 22.40): its FUSED
  prologue logic only, ported into the c320 kernel A as a template flag FUSED (false = SOURCE A
  byte for byte): the request's raw bf16 q row (2 KB) rides item 0's page-index cp.async group;
  after the first tiles are issued every CTA builds its q MMA fragments in registers (scaleless
  RMSNorm eps 1e-5, times qk scale handling exactly as _pre_attention_kernel does for full layers:
  no RoPE, e4m3 cast, the same column mapping the non-fused path reads from smem); the CTA that
  owns the item containing key seq-1 also loads the raw new k/v heads (512 B) and cache_loc[0],
  builds the new row's e4m3 bytes (k / k_scale, v / v_scale) during the same window, overwrites
  that key's slot in the landed ring stage in smem (never reads it back through cp.async) and
  stores it to k/v cache at cache_loc (one extra __syncthreads per launch). kc/vc become non-const
  in this module only; master's modules are not touched. SOURCE D = splitkv-pdl-attn-to-combine
  (branch hyp/muse_attention_decode/splitkv-pdl-attn-to-combine, impl 9b85387, job
  113cbf3093dd40b9, written for exactly this v1 A/B kernel pair; at B=16 its shape went 122.79 ->
  120.56 us): griddepcontrol.launch_dependents in kernel A and griddepcontrol.wait at the top of
  the combine kernel's partial reads, combine launched with cudaLaunchKernelEx +
  cudaLaunchAttributeProgrammaticStreamSerialization; a failed cudaLaunchKernelEx clears its error
  and falls back to the plain launch (as in the source). Instead of the source's static g_pdl
  toggle, compile the PDL and plain launches as separate host entry points (no mutable flag read
  per call). VARIANTS at FD1, all in the one c320 module: V0 = master [pre-attn + two-level]
  (incumbent); V1 = [pre-attn + A<320> + combine] (SOURCE A alone); V2 = V1 + PDL A->combine
  (A+D); V3 = [A<320,FUSED> + combine] (A+C, 2 nodes); V4 = V3 + PDL (A+C+D). GATE: correctness
  (i) and two-state replay (ii) of GATE A for every variant (the FUSED ones also compare the
  stored new K/V rows with master _pre_attention_kernel: <= 3 of 128 bytes differ, rel L2 < 2e-2,
  other rows unchanged, as the fused source did); speed (iii) all five variants interleaved in one
  pass (cold L2, empty subtracted, 41 reps), then the pass-1 best re-timed head to head with V0
  and V1 (31 reps); route the best only if best / V0 <= 0.98; print every variant's number and
  'PATH CHOSEN: full decode_1 [chain]: V<k> (<sources>) ...'. Every other class stays on master
  byte for byte. PHASES AFTER MASTER: c320 module with FUSED and PDL variants ~50 s (deadline 120
  s); correctness of 4 variants ~10 s; gate ~15 s (deadline 60 s for both). ~75 s added. IMPORT
  BUDGET (no prepare() on this problem: everything runs at import and counts toward the 15-min
  run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s each). Master's
  import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1, _ps_setup, _skf_setup,
  diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its own thresholds. Every
  added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock and is SKIPPED (master
  kept for its classes, PATH CHOSEN says so) if the clock is past 420 s when it would start or
  past its own deadline when it finishes. RULES (every path): CUDA-graph capturable; kernel_fn
  does no host sync, no allocation sized by input values (shape/config only), no data-dependent
  Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached
  from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key of the context
  attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates
  them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (the c320 path has no tickets; its workspaces are torch.empty per call, sized
  from page_table.shape[1] and batch). Route tables are frozen (MappingProxyType / module bools
  set once at import) before any capture.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "V1 alone is 19.49 us on the judge; folding the 1.3 us pre-attention node (C measured -1.24 us on the two-level kernel) and overlapping the combine prologue with A's tail (D) targets ~16.5-17.5 us: -5 to -6 us x 16 on the 905 us decode_1 load, ~+1.5-1.8% headline"
  - shapes: "*layer=sliding *"
    direction: flat
    why: master byte for byte
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: B>1 untouched (v2 stream-K + skF gate)
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-b1-chain-c320-fused-pre-pdl

## Change

COMBINATION 2 of 3, a conjectural B=1 launch-chain pairing (not the top rows). Built on current master 7e12891 (blob 5cf8fae). Full decode_1 is the highest-leverage full row (weight 16 in the smallest load, 905 us: ~0.29% headline per us). On it, each measured design leaves 1.0-1.4 us per graph node of non-kernel time: master two-level 20.4 us of kernels in a 22.4 us shape (2 nodes), chunk320 16.8 us in 19.49 (3 nodes), fused two-level 20.3 in 21.6 (1 node). This combination puts the fastest kernel design (chunk320) on the fewest effective nodes by stitching two sources that each removed one node boundary elsewhere. SOURCE A = full-decode1-cuda-splitkv-chunk320 (branch hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job 8d4771feca9b4e18: full decode_1 19.49 us = splitkv_attn_kernel<320> 11.0 + splitkv_combine_kernel<320,16> 4.5 + _pre_attention_kernel 1.3; master's two-level stream-K is 22.40 us in 252abb7f and 22.53 in 917a9689, so 0.87x of master on the same row, paired vs SGLang 1.604 vs 1.393/1.395). It was never compared with the two-level kernel: it harvested neutral on an older base and s3 landed after it. Master's _SPLITKV_CUDA is byte-identical to that run's base, so the CUDA ports cleanly. Port: the templated splitkv_attn_kernel<CHUNK> and splitkv_combine_kernel<CHUNK, NW> with launch_attn<320> / launch_combine<320,16> (512-thread combine), the strided issue_meta loop, the __host__ __device__ constexpr smem helpers (audit fix), instantiated ONLY at <320> and <320,16>, as a NEW load_inline module muse_full_decode_c320 (TORCH_CUDA_ARCH_LIST 9.0a, -O3 -lineinfo, -std=c++20 in extra_cuda_cflags, binaryVersion == 90 check, print regs/local/smem). Master's v1 module and its <1024> code stay untouched. Python: _full_decode_c320(data, q, qkv) = master _pre_attention_kernel (unpaired, exactly as master's batch-1 branch) then c320 A + combine, max_chunks = ceil(page_table.shape[1] / 320), grid 2 x SM. Gate class FD1 = config window == 0, q_len 1, batch 1, tp8, d 64, page 1 (the same predicate as master's _shape_gate at B=1). GATE A (FD1): (i) correctness: the chunk320 run's batch-1 cases (lens [80000] and [5000], kv 1 and 0.25, grid device and 3) plus 1 / 63 / 64 / 4097 keys, rel L2 < 2e-2 vs master's FP32 _splitkv_reference AND < 5e-3 vs master's two-level output on the same inputs, finite, cache rows equal to master's pre-attention rows (same kernel); (ii) two-state alternating capture-once replay (A B B A A B, refilled inputs, NaN-poisoned outputs) at 80000 and 4097 keys; (iii) speed: interleaved cold-L2 CUDA-graph replay (256 MB flush, empty graph subtracted, 41 reps, median) of [pre-attn + c320 A + combine] vs master's [pre-attn + two-level <4,true,true> grid SM] at B=1 x 80000; route only if c320 / master <= 0.98 (expected ~0.87). Print 'PATH CHOSEN: full decode_1 [A chunk320]: <route|master two-level> (c320 X us vs two-level Y us, ratio R)'. SOURCE C = full-decode1-sk-two-level-fused-preattn (branch hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn, impl 2236339, job 2038a530edd04309; its gate 20.74 vs 21.98 us = 1.060x, judged 21.60 vs 22.40): its FUSED prologue logic only, ported into the c320 kernel A as a template flag FUSED (false = SOURCE A byte for byte): the request's raw bf16 q row (2 KB) rides item 0's page-index cp.async group; after the first tiles are issued every CTA builds its q MMA fragments in registers (scaleless RMSNorm eps 1e-5, times qk scale handling exactly as _pre_attention_kernel does for full layers: no RoPE, e4m3 cast, the same column mapping the non-fused path reads from smem); the CTA that owns the item containing key seq-1 also loads the raw new k/v heads (512 B) and cache_loc[0], builds the new row's e4m3 bytes (k / k_scale, v / v_scale) during the same window, overwrites that key's slot in the landed ring stage in smem (never reads it back through cp.async) and stores it to k/v cache at cache_loc (one extra __syncthreads per launch). kc/vc become non-const in this module only; master's modules are not touched. SOURCE D = splitkv-pdl-attn-to-combine (branch hyp/muse_attention_decode/splitkv-pdl-attn-to-combine, impl 9b85387, job 113cbf3093dd40b9, written for exactly this v1 A/B kernel pair; at B=16 its shape went 122.79 -> 120.56 us): griddepcontrol.launch_dependents in kernel A and griddepcontrol.wait at the top of the combine kernel's partial reads, combine launched with cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization; a failed cudaLaunchKernelEx clears its error and falls back to the plain launch (as in the source). Instead of the source's static g_pdl toggle, compile the PDL and plain launches as separate host entry points (no mutable flag read per call). VARIANTS at FD1, all in the one c320 module: V0 = master [pre-attn + two-level] (incumbent); V1 = [pre-attn + A<320> + combine] (SOURCE A alone); V2 = V1 + PDL A->combine (A+D); V3 = [A<320,FUSED> + combine] (A+C, 2 nodes); V4 = V3 + PDL (A+C+D). GATE: correctness (i) and two-state replay (ii) of GATE A for every variant (the FUSED ones also compare the stored new K/V rows with master _pre_attention_kernel: <= 3 of 128 bytes differ, rel L2 < 2e-2, other rows unchanged, as the fused source did); speed (iii) all five variants interleaved in one pass (cold L2, empty subtracted, 41 reps), then the pass-1 best re-timed head to head with V0 and V1 (31 reps); route the best only if best / V0 <= 0.98; print every variant's number and 'PATH CHOSEN: full decode_1 [chain]: V<k> (<sources>) ...'. Every other class stays on master byte for byte. PHASES AFTER MASTER: c320 module with FUSED and PDL variants ~50 s (deadline 120 s); correctness of 4 variants ~10 s; gate ~15 s (deadline 60 s for both). ~75 s added. IMPORT BUDGET (no prepare() on this problem: everything runs at import and counts toward the 15-min run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s each). Master's import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1, _ps_setup, _skf_setup, diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its own thresholds. Every added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock and is SKIPPED (master kept for its classes, PATH CHOSEN says so) if the clock is past 420 s when it would start or past its own deadline when it finishes. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key of the context attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (the c320 path has no tickets; its workspaces are torch.empty per call, sized from page_table.shape[1] and batch). Route tables are frozen (MappingProxyType / module bools set once at import) before any capture.

## Expected

- `*layer=full load=decode_1 *` faster: V1 alone is 19.49 us on the judge; folding the 1.3 us pre-attention node (C measured -1.24 us on the two-level kernel) and overlapping the combine prologue with A's tail (D) targets ~16.5-17.5 us: -5 to -6 us x 16 on the 905 us decode_1 load, ~+1.5-1.8% headline
- `*layer=sliding *` flat: master byte for byte
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: B>1 untouched (v2 stream-K + skF gate)
- `*load=verify_* *` flat: untouched

## Risk

Medium risk: SOURCE A ran clean on the judge; SOURCE C's prologue ran clean on the judge inside the two-level kernel but here moves into a different kernel (new code, not a verbatim port); SOURCE D ran clean on the judge on this kernel pair. No source has an unexplained fault. Every fused/PDL variant is a separate instantiation behind the gate, with V1 as the always-available intermediate, and master V0 as the fallback.

## Conjecture

The interaction is a launch chain that crosses two ports. SOURCE D's PDL was flat-to-slightly-positive at B=16 because one node boundary is ~1% of a 120 us shape; SOURCE C's fusion saved 1.24 us on the two-level kernel; PDL from the Triton pre-attention into a CUDA kernel LOST (+0.54 us attribute, +3.84 us with the trigger, full-decode1-sk-pdl-from-preattn), so the only safe PDL edge is CUDA->CUDA. Claim: at B=1 on the c320 design the two node boundaries are ~2.7 us of a 19.5 us shape (13%), so removing the pre-attention node by fusion and hiding the A->combine boundary by PDL are worth more together than either source showed alone, and the order matters: the fused prologue lengthens A's start by ~0.3 us of register q build that the 250-CTA grid hides behind its first cp.async tiles, while PDL lets the 16-CTA combine's launch and its qkv-gate read overlap A's tail. Counter-claim worth learning: with 250 CTAs at 2/SM there is no idle SM for the PDL'd combine to occupy until A drains, so V2 ~ V1 and the whole gain is C's. Readout: the printed V0..V4 gate numbers (prepare ranks variants reliably; judged numbers run 20-25% above prepare) and the judged profile on full decode_1: V3/V4 show no _pre_attention_kernel row; shape minus kernel sum should fall from ~2.7 us (V1) toward ~1 us (V4). If V2 - V1 ~ 0 and V4 - V3 ~ 0, PDL is dead at B=1 on this pair; if V3 > V1, the fused prologue costs more than a node at 250 CTAs (each CTA rebuilds q, 250x the 2 KB read).
