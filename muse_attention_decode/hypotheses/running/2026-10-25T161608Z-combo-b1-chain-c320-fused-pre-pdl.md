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
review: proceed
status: running
review_at: 2026-10-25T162106Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T162106Z
base_sha: c20f01f78876c85beb295ad153dccfe3078c3ad4
base_blob: 5cf8fae31e5b84c36efeeb4569763eb9b06e1c8a
branch: hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl
worktree: .worktrees/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T162106Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a scored submit, with three build fixes. The fused K row must be RMSNorm'd before /k_scale. A compile failure of the FUSED instantiation must not take V1 down with it. PDL's success or failure must be decided once at import, with no per-call fallback. The in-import V0..V4 gate makes the single judged run readable. The conjecture can only be tested on prepare-time gate numbers: the judge measures one routed variant.

### Justification
Mechanism is plausible. Master full decode_1 is 22.4 us shape for 20.4 us of kernels (sk two-level 19.0 + pre-attn 1.4; SoL 4.3 us), so ~2 us is node and launch overhead. chunk320 measured 19.49 us shape (A 11.0 + combine 4.5 + pre 1.3). I verified that master's _SPLITKV_CUDA is byte-identical (md5) to the bases of 183eb55, 9b85387 and 2236339, so A and D port onto known code. C removed the pre-attn node on the two-level kernel: judged 21.60 vs 22.40, so about -0.8 us judged and -1.24 us at gate. D has no B=1 evidence: in its source run decode_1 was still on FA3, and B=16 moved only 122.79 to 120.56, within CI. The weakest link is the target of 16.5-17.5 us. C's judged saving was ~0.8 us, not 1.24. In the PDL source, launch_dependents sits AFTER the partial writes, so PDL hides only the combine's launch latency (sub-us), not any of its 4.5 us of work. A realistic V4 is ~18.0-18.8 us judged. That is still a clear decode_1 gain over 22.4 (~-3.6 to -4.4 us x16, roughly +1.0-1.3% headline), but below the stated +1.5-1.8%. A second weak link is that V1 vs V0 was never measured head to head: 19.49 came from an older base and 22.4 from master. The 0.98 gate covers this. The real remaining B=1 cost is the 4.5 us, 16-CTA combine, and neither C nor D touches it. Expected directions are consistent: only FD1 changes, and every other class stays on master byte for byte. Headline readability is poor because full decode_32 noise is large: the untouched decode_32 row moved -8% in the chunk320 run and -5% in the PDL run. Read the full decode_1 row and the PATH CHOSEN / V0..V4 lines, not the headline. Conjecture testability: the judge runs only the routed variant. The V2-V1 and V4-V3 interaction terms therefore come only from the prepare gate pass, where the differences (~0.3-1 us on ~16 us) sit near prepare noise. The builder should print the interaction term (V1-V4) - [(V1-V2)+(V1-V3)] with the per-variant spreads (IQR or p25/p75), so the claim of super-additivity is confirmed or refuted from the log. On the judged profile, also check the expected sequence for V3/V4: no _pre_attention_kernel row, and shape minus kernel sum drops from ~2.7 us. Can the new fused prologue be built correctly without a check run? Yes, conditionally. The import gate checks rel L2 vs the FP32 reference and vs master, compares the stored K/V row bytes (<=3/128 differ, other rows unchanged) and runs the two-state A B B A A B replay. These catch numerical and cache-store errors and fall back to V1/V2 or V0. The judge therefore never sees a wrong FUSED path, but a bug costs the conjecture. The spec as written has one definite bug: it says the new row is built as 'k / k_scale, v / v_scale'. Master _pre_attention_kernel applies the scaleless RMSNorm (eps 1e-5, over the head's 64 entries, per KV head) to K before dividing by k_scale; V gets no norm. Built literally, the stored-row check would fail and V3/V4 would be dropped. Port source C's K path, not the paragraph. Other fused-prologue hazards. (a) The key seq-1 overwrite must land after cp_wait for that stage and before the QK MMA of that tile, for both the K and V rings, with the __syncthreads placed so that no warp is still reading the stage. (b) With one item per CTA (250 items on 264 CTAs), the q build is not hidden behind multiple items as it was in two-level, so V3 > V1 is a real possibility, which the conjecture already admits. (c) The e4m3 cast should use RTNE with saturation (__NV_SATFINITE) to match Triton's float8e4nv on in-range values. Import budget: the job wall was ~489 s on master, so ~75 s more is fine. The 420 s / 120 s / 60 s deadlines bound the added phases to at most ~180 s past 420 s. Since a hung nvcc would not be cut, the worst case stays under the 900 s cap only if master's own benchmark tail stays under ~300 s, which it has.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (landed): this is V0, the incumbent the gate compares against. Its B=1 two-level ticket path is the whole denominator, so the routing decision depends on how fast it is in prepare. No code interaction, because c320 has no tickets and V0 is untouched.
- full-decode-cuda-gather-splitkv (landed): supplies _SPLITKV_CUDA, which A and D modify. It is byte-identical to the sources' bases (verified), so the ports are not confounded by drift.
- fused-preattention-triton / paired-heads-launch-cleanup-ab-tune-batch-gate (landed): V1/V2 use master's unpaired _pre_attention_kernel at B=1, and V3/V4 must reproduce its K-norm, V-scale and q e4m3 semantics exactly. Any mismatch shows as a stored-row gate failure, not as a speed effect.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (landed skF FUSED_PRE at 1<B<=64): a different fused prologue in a different kernel and module. No runtime interaction at B=1. It matters only as a second, independent implementation of the same fusion that a builder might copy from by mistake (it is the B>1 layout).
- Judge noise on full decode_32 / decode_16 (untouched rows that moved -5% to -8% in the chunk320 and PDL runs): the headline can swing either way, independent of this change.

### Interference and A/B needs
No companion is warranted. Every variant is a separate instantiation behind one config-only FD1 gate. The import gate times V0..V4 interleaved in one cold-L2 pass and prints all five, and PATH CHOSEN names the routed one. A judged gain or loss on full decode_1 is therefore attributable to the named variant against the V0 incumbent, without an extra run. An ab-revert of any landed hypothesis would not change what is measured: master's B=1 path is the comparator, and no landed code is modified. A control that isolates C or D is already built in as V1/V2/V3, at the cost of the conjecture being read from prepare numbers rather than judged ones. Buying a judged V1-only run to attribute it would spend a second scored submit to recover what chunk320's 19.49 us already gives approximately. Process-global flag policy: the hypothesis already replaces the source's static g_pdl toggle with separate PDL and plain host entry points, which complies. Two related requirements: (1) remove the source's per-call 'cudaLaunchKernelEx failed -> cudaGetLastError -> plain launch' fallback from the PDL entry point, because a failed launch inside capture invalidates the capture anyway. Decide PDL availability once at import, via the gate's capture/replay test, and route V2/V4 only if it passed; the PDL entry point then just C10_CUDA_CHECKs the Ex launch. (2) Move the launch_attn 'static bool attr_set' cudaFuncSetAttribute to an explicit import-time init for both A<320,false> and A<320,true>, so nothing in the call path mutates process state, even once. Neither change needs a cell dropped.

### Open questions
- Build isolation: if the A<320,FUSED=true> instantiation fails nvcc, the single muse_full_decode_c320 module fails, and V1/V2 are lost with it. Either guard FUSED with a preprocessor macro and, on a build exception, rebuild once without it while the IMPORT_T0 clock is < 420 s, or compile-check the CUDA source locally (nvcc -arch=sm_90a -std=c++20) before submitting. The latter is cheap and should be done regardless, since there is no check run today.
- Does the combine's qkv-gate read actually move above griddepcontrol.wait (as the Conjecture claims), or stay below it (as in the PDL source)? State which. Only the former gives D anything beyond launch latency.
- launch_dependents placement in A: the source triggers after the partial writes (end of work), so the only overlap is combine launch latency. If an earlier trigger is tried, it is safe for correctness (wait blocks until A fully completes and flushes) but can park 512-thread combine CTAs on SMs. With 264 CTAs at 2/SM there is no free slot except the 14 idle-CTA slots. Keep the source's placement unless the gate shows a reason.
- Harvest: compare full decode_1 against 22.4 us (master) and 19.49 us (chunk320 alone). A pass is <= ~19 us with the routed variant named. Read V0..V4 and the printed interaction term from the import log to confirm or refute super-additivity. A judged result near 19.5 us with V3/V4 routed means C and D added nothing on c320.

