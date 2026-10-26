---
problem: muse_attention_decode
slug: sliding-decode-sk-s3-fused-preattn-rope
created: 2026-10-25T110716Z
base: master
area: sliding-layer attention (decode, stream-K windowed kernel, pre-attention fold)
change: >
  Stage 3 of the sliding-decode stream-K line, built on the branch of
  restack-sliding-decode-sk-window-on-cluster-master (master eb1b68b + the parent's TOK=1 windowed
  sk kernel routed at sliding decode B=17..32), whatever its score. The parent's run left the
  separate pre-attention node as the remaining cost. Routed B=32 profile: sk kernel 13.9 us +
  _pre_attention_kernel 1.9 us = 15.8 us of kernels in a 17.38 us shape, i.e. ~3.5 us of the row
  is the second launch plus its graph-node gap. At B=16 the pair lost to Triton by exactly that
  node: sk alone 13.1 us net, pre-attention alone 3.7 us net, pair 16.2 vs Triton 16.3 us. Its
  review named the next step: fold pre-attention into the sk prologue, as master's FUSED_PRE (skF,
  splitkv_attn_sk_fused_kernel, module muse_decode_skf_v1) already does for full decode. CHANGE:
  give splitkv_attn_sk_fused_kernel one template flag SLIDE (false = the landed full-decode
  kernel, byte-identical: confirm its regs/local/smem line matches master's [skF] attrs).
  SLIDE=true adds two things. (1) Windowed rows, as the TOK=1 PSEUDO kernel does: per request sp =
  cache_seqlens[b], lo_b = max(0, sp - 2048), cnt_b = sp - lo_b, built with load_prefix_pseudo<1>
  (or the same scan inline) into a LO table (Lay3 grows by MAXB ints). The page-table offset adds
  lo_b, and the masks use cnt_b. The owner of the new token is the CTA holding request b's LAST
  windowed tile (key sp - 1); _skf_owner_check must use windowed tile counts. (2) RoPE
  (interleaved GPT-J pairs (2i, 2i+1), angle = positions[b] * inv_freq[i], inv_freq from master's
  _inv_freq(theta, 64)). (2a) In the prologue's new-row store, the k head is RMS-normalized, then
  rotated, then divided by k_scale; each lane owns dims 2*lane and 2*lane+1, so it owns exactly
  one RoPE pair. v is unchanged. (2b) In the segment's q build, after the scaleless RMSNorm and
  before the e4m3 rounding, rotate each pair the same way. Use accurate sincosf; the self-check
  compares by rel L2 against an fp32 reference, never bytewise against Triton (LESSONS: Triton's
  sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad). New args: positions (int64 [B]) and inv_freq
  (f32 [32]); both are read on the full-decode path only when SLIDE. Host side: an skf_attn
  variant flag, plus _sdf_launch(data, qkv, grid) that mirrors _skf_launch with the _ps_ticket
  self-resetting array. Grid specs p4/p8/SM per band. Route by an import-time frozen table
  (MappingProxyType) read in kernel_fn by the same config-only _sd_gate predicate: a band in
  _SDF_ROUTE takes the ONE fused launch, else stage 2's pair if that band is in _SD_ROUTE, else
  master (cluster/Triton). GATE per band (B=1 record only, B=2..16 at 16, B=17..32 at 32), at
  batch x 80000. Interleaved cold-L2 graph replay with the empty graph subtracted; time [fused
  SLIDE], [pre-attn + sk TOK=1] and the band's master incumbent (Triton at 16, cluster C=4 at 32,
  cluster C=16 at 1). Route only if fused beats the faster of the other two by >= 1.03x, and print
  all three. SELF-CHECK vs fp32 windowed reference with RoPE: lens 0, 1, 63, 2047, 2048, 2049,
  80000, ragged B=16/32, kv 0.25, grids 3/7/p4/SM. Read the new cache row back and compare it by
  rel L2 with _pre_attention_kernel's row. Run the two-state alternating replay (capture kernel_fn
  once per state, refill, NaN-poison, compare each). Any miss leaves that band on stage 2/master
  with a PATH CHOSEN line. BUILD as the landed TUs: TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo,
  -std=c++20 if flags are overridden, in-process canary (B=1 lens 1), sync + print after each
  self-check launch. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values, no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs (positions and the
  cache row are computed per call); KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out;
  every key in each window [p-2047, p] attended; no child processes; no module-global device
  pointers into a state.
expected:
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: incumbent (stage 2 restack) 17.38 us shape = sk 13.9 + pre-attn 1.9 + ~1.6 us node gap; one launch removes pre-attn and the gap, M2 shape <= 15.5 us (floor 3.3 us); vs master cluster C=4 20.6 us
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: Triton 15.4 us shape (kernel 12.8); stage 2 pair lost by the pre-attn node (16.2 vs 16.3 us net, sk alone 13.1, pre 3.7); folded, the kernel alone should beat Triton by >= 1.03x if the sk body runs ~12 us; floor 1.6 us
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: 32 tiles only; record-only band, cluster C=16 (11.9 us shape) kept
  - shapes: "*layer=full *"
    direction: flat
    why: SLIDE=false instantiation of skF is the landed kernel; PSEUDO=false sk untouched
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: restack-sliding-decode-sk-window-on-cluster-master
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "sliding decode (q_len 1, page 1, tp8, 2<=B<=32): stage-2 [_pre_attention_kernel + splitkv_attn_sk_kernel TOK=1] pair at B=17..32 and master Triton _sliding_fused_kernel at B=2..16"
review: proceed
status: queued
review_at: 2026-10-26T220207Z
review_companions: null
---

# sliding-decode-sk-s3-fused-preattn-rope

## Change

Stage 3 of the sliding-decode stream-K line, built on the branch of restack-sliding-decode-sk-window-on-cluster-master (master eb1b68b + the parent's TOK=1 windowed sk kernel routed at sliding decode B=17..32), whatever its score. The parent's run left the separate pre-attention node as the remaining cost. Routed B=32 profile: sk kernel 13.9 us + _pre_attention_kernel 1.9 us = 15.8 us of kernels in a 17.38 us shape, i.e. ~3.5 us of the row is the second launch plus its graph-node gap. At B=16 the pair lost to Triton by exactly that node: sk alone 13.1 us net, pre-attention alone 3.7 us net, pair 16.2 vs Triton 16.3 us. Its review named the next step: fold pre-attention into the sk prologue, as master's FUSED_PRE (skF, splitkv_attn_sk_fused_kernel, module muse_decode_skf_v1) already does for full decode. CHANGE: give splitkv_attn_sk_fused_kernel one template flag SLIDE (false = the landed full-decode kernel, byte-identical: confirm its regs/local/smem line matches master's [skF] attrs). SLIDE=true adds two things. (1) Windowed rows, as the TOK=1 PSEUDO kernel does: per request sp = cache_seqlens[b], lo_b = max(0, sp - 2048), cnt_b = sp - lo_b, built with load_prefix_pseudo<1> (or the same scan inline) into a LO table (Lay3 grows by MAXB ints). The page-table offset adds lo_b, and the masks use cnt_b. The owner of the new token is the CTA holding request b's LAST windowed tile (key sp - 1); _skf_owner_check must use windowed tile counts. (2) RoPE (interleaved GPT-J pairs (2i, 2i+1), angle = positions[b] * inv_freq[i], inv_freq from master's _inv_freq(theta, 64)). (2a) In the prologue's new-row store, the k head is RMS-normalized, then rotated, then divided by k_scale; each lane owns dims 2*lane and 2*lane+1, so it owns exactly one RoPE pair. v is unchanged. (2b) In the segment's q build, after the scaleless RMSNorm and before the e4m3 rounding, rotate each pair the same way. Use accurate sincosf; the self-check compares by rel L2 against an fp32 reference, never bytewise against Triton (LESSONS: Triton's sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad). New args: positions (int64 [B]) and inv_freq (f32 [32]); both are read on the full-decode path only when SLIDE. Host side: an skf_attn variant flag, plus _sdf_launch(data, qkv, grid) that mirrors _skf_launch with the _ps_ticket self-resetting array. Grid specs p4/p8/SM per band. Route by an import-time frozen table (MappingProxyType) read in kernel_fn by the same config-only _sd_gate predicate: a band in _SDF_ROUTE takes the ONE fused launch, else stage 2's pair if that band is in _SD_ROUTE, else master (cluster/Triton). GATE per band (B=1 record only, B=2..16 at 16, B=17..32 at 32), at batch x 80000. Interleaved cold-L2 graph replay with the empty graph subtracted; time [fused SLIDE], [pre-attn + sk TOK=1] and the band's master incumbent (Triton at 16, cluster C=4 at 32, cluster C=16 at 1). Route only if fused beats the faster of the other two by >= 1.03x, and print all three. SELF-CHECK vs fp32 windowed reference with RoPE: lens 0, 1, 63, 2047, 2048, 2049, 80000, ragged B=16/32, kv 0.25, grids 3/7/p4/SM. Read the new cache row back and compare it by rel L2 with _pre_attention_kernel's row. Run the two-state alternating replay (capture kernel_fn once per state, refill, NaN-poison, compare each). Any miss leaves that band on stage 2/master with a PATH CHOSEN line. BUILD as the landed TUs: TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 if flags are overridden, in-process canary (B=1 lens 1), sync + print after each self-check launch. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values, no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs (positions and the cache row are computed per call); KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window [p-2047, p] attended; no child processes; no module-global device pointers into a state.

## Expected

- `*layer=sliding load=decode_32 *` faster: incumbent (stage 2 restack) 17.38 us shape = sk 13.9 + pre-attn 1.9 + ~1.6 us node gap; one launch removes pre-attn and the gap, M2 shape <= 15.5 us (floor 3.3 us); vs master cluster C=4 20.6 us
- `*layer=sliding load=decode_16 *` faster: Triton 15.4 us shape (kernel 12.8); stage 2 pair lost by the pre-attn node (16.2 vs 16.3 us net, sk alone 13.1, pre 3.7); folded, the kernel alone should beat Triton by >= 1.03x if the sk body runs ~12 us; floor 1.6 us
- `*layer=sliding load=decode_1 *` flat: 32 tiles only; record-only band, cluster C=16 (11.9 us shape) kept
- `*layer=full *` flat: SLIDE=false instantiation of skF is the landed kernel; PSEUDO=false sk untouched
- `*load=verify_*` flat: outside the gate

## Risk

Kill: if fused B=16 is not >= 1.03x faster than Triton and B=32 not >= 1.03x faster than the stage-2 pair in the printed gate, the line has reached its node-count floor and the next stage must speed up the sk body itself (0.64 TB/s at B=16 is latency-bound: 4 tiles per CTA). The new RoPE code is plain fp32 FMA plus sincosf, with no new PTX forms, so the fault risk is the new LO-table indexing. Keep the canary and per-launch prints. Run the first measurement as an okc check. Worth ~+1-2% headline over the restack (two weight-46 rows).

## Milestones

Rewrite stage 3 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): stage-2 [_pre_attention_kernel + splitkv_attn_sk_kernel TOK=1] pair at B=17..32 and master Triton _sliding_fused_kernel at B=2..16`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-26T220207Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as an okc check with no companions: the design is sound, gated per band and readable from the decode_16/32 sliding rows. It is the same FUSED_PRE fold DIAGNOSIS lists as do-not-try for full decode (+3.1/+3.2 us at full B=16, ~0 at B=32), so expect roughly 1.0-1.15x per row and +0.3-1.0% headline. Two build traps (load_prefix_pseudo has an empty body in the skF TU; the dynamic-smem attribute is set for one kernel only) and a missing window-edge self-check must be fixed before dispatch.

### Justification
Removing the separate _pre_attention_kernel node is a real lever on these latency-bound rows (sliding decode_32 17.80 us = sk 14.1 + pre 1.9; decode_16 15.24 us Triton 12.6), but the fold adds in-kernel cost (skF measured +3.2 us at full B=16). Gate budgets: B=32 fold <= 2.4 us, B=16 <= 2.1-2.7 us: each band close to a coin flip. M2 (decode_32 <= 15.5 us) is the operative milestone; M3 is unreachable at these sizes. The difference from full-decode-splitkv-s3-fused-preattn is real at B=2..16 (bar is Triton's single launch).

### Confounders on the current kernel
- restack-sliding-decode-sk-window-on-cluster-master (parent): its pair is the B=17..32 incumbent; read _SD_ROUTE and _SDF_ROUTE PATH CHOSEN lines
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: templating skF can perturb SLIDE=false codegen (125/128 regs)
- sliding-decode1-cluster-per-kvhead (master head): build from current master 5d528bf3, not the stale restack branch
- fused-preattention-triton: Triton sin/cos approximate at ~80000 rad; use rel < 5e-2 vs Triton, byte match info only
- sliding-deep-splits-on-cuda-splitkv + sliding-fused-triton-decode1-gate: the Triton single-launch incumbent at B=2..16
- shared self-resetting _ps_ticket: SDF setup needs a finally re-zero

### Interference and A/B needs
No companion: rewrite stage with its own in-process A/B per band; gain attributable from the decode_16/32 sliding rows plus [sdf-gate] lines; print [fused] - [sk alone] per band as the fold cost.

### Open questions
- BUILDER (blocking): load_prefix_pseudo is empty in the skF TU (no -DSK_PSEUDO): inline the scan under if constexpr (SLIDE) or add the define
- BUILDER (blocking): set the dynamic-smem attribute for both skF instantiations
- BUILDER: self-check on _slide_synthetic with edge keys, B=4 and B=16 cases
- BUILDER: load positions / inv_freq in the prologue or a smem cos/sin table
- BUILDER: __threadfence only on the owner branch that stored
- BUILDER: two-round gate (>= 1.03x vs Triton at B=2..16; vs the routed pair at B=17..32); fp32 accurate-RoPE row reference
- BUILDER: skip timing the B=1 band; own try/except freezing _SDF_ROUTE; re-zero _ps_ticket in finally
- HARVEST: judge on sliding decode_16/32 rows and [sdf-gate] lines vs master 4b608df595b046fb

