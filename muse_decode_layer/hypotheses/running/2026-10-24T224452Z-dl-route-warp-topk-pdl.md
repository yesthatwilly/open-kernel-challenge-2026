---
problem: muse_decode_layer
slug: dl-route-warp-topk-pdl
created: 2026-10-24T224452Z
base: master
area: routing top-k + MoE align (kernel R, T<=32)
change: >
  Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32
  (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes, nothing else: (a) top-8
  without block barriers: replace the 8-round loop (warp_max_u64 + s_kred + 2 __syncthreads per
  round) with a per-warp top-8: each warp runs 8 rounds of warp_max_u64 over its 32 keys with a
  lane-local taken flag (no block sync), lane r writes its round-r winner to s_cand[warp*8+r]; one
  __syncthreads; warp 0 then picks the global top-8 of the 64 candidates (each lane holds 2, 8
  rounds of warp_max_u64) and writes s_sel[0..7]; one __syncthreads. route_key, tie order (lower
  id wins), weights s/(sum+1e-15), topk_ids, topk_w and local_ids writes unchanged. (b) PDL
  between P and R: in dl_pooler_router_kernel add griddepcontrol.launch_dependents after the main
  K loop of every CTA (before the split-K partial store), and launch R with cudaLaunchKernelEx and
  the programmatic serialization attribute when dl["pdl"] is 1; in R, load norm_w (its 16 values
  per thread) and bias[e] into registers BEFORE griddepcontrol.wait and read ei/logits only after
  it. Also hoist the pad loop: compute each padding slot from s_seg0/s_seg1/s_count by a binary
  search over 32 experts instead of the linear scan. Self-check: the existing _dl_self_check cases
  (routing, near-tie, no-local-slot, align structure, Marlin y, graph replay) must pass unchanged;
  print R CUDA-event time with pdl 0/1 for T 1, 8, 32.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "R 7.6 -> ~3.5 us plus launch overlap with P tail: ~4-5 us of 132.7/142.6 (+3%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: ~4 us of 186.5 (+2%)
  - shapes: "*load=decode_16 *"
    direction: faster
    why: ~4 us of 217 (+1.5-2%)
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~4 us of 247 (+1.5%)
  - shapes: "*load=verify_16 *"
    direction: flat
    why: outside gate
  - shapes: "*load=verify_32 *"
    direction: flat
    why: outside gate
risk: medium
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T010823Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T010823Z
base_sha: 9d09de0cc3ef315d6ef3aab918058321b698da3f
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/dl-route-warp-topk-pdl
worktree: .worktrees/muse_decode_layer/dl-route-warp-topk-pdl
mode: null
restack_of: null
stale_base: null
---

# dl-route-warp-topk-pdl

## Change

Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32 (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes, nothing else: (a) top-8 without block barriers: replace the 8-round loop (warp_max_u64 + s_kred + 2 __syncthreads per round) with a per-warp top-8: each warp runs 8 rounds of warp_max_u64 over its 32 keys with a lane-local taken flag (no block sync), lane r writes its round-r winner to s_cand[warp*8+r]; one __syncthreads; warp 0 then picks the global top-8 of the 64 candidates (each lane holds 2, 8 rounds of warp_max_u64) and writes s_sel[0..7]; one __syncthreads. route_key, tie order (lower id wins), weights s/(sum+1e-15), topk_ids, topk_w and local_ids writes unchanged. (b) PDL between P and R: in dl_pooler_router_kernel add griddepcontrol.launch_dependents after the main K loop of every CTA (before the split-K partial store), and launch R with cudaLaunchKernelEx and the programmatic serialization attribute when dl["pdl"] is 1; in R, load norm_w (its 16 values per thread) and bias[e] into registers BEFORE griddepcontrol.wait and read ei/logits only after it. Also hoist the pad loop: compute each padding slot from s_seg0/s_seg1/s_count by a binary search over 32 experts instead of the linear scan. Self-check: the existing _dl_self_check cases (routing, near-tie, no-local-slot, align structure, Marlin y, graph replay) must pass unchanged; print R CUDA-event time with pdl 0/1 for T 1, 8, 32.

## Expected

- `*load=decode_1 *` faster: R 7.6 -> ~3.5 us plus launch overlap with P tail: ~4-5 us of 132.7/142.6 (+3%)
- `*load=verify_1 *` faster: ~4 us of 186.5 (+2%)
- `*load=decode_16 *` faster: ~4 us of 217 (+1.5-2%)
- `*load=decode_32 *` faster: ~4 us of 247 (+1.5%)
- `*load=verify_16 *` flat: outside gate
- `*load=verify_32 *` flat: outside gate

## Risk

Medium: a wrong tie order fails routing checks; the near-tie self-check case covers it. The profiler row for R will include its PDL wait when pdl=1, so judge on the printed CUDA-event times and the shape totals.

## Adversarial review

Reviewed 2026-10-25T010823Z by an adversarial subagent (model opus).

**Verdict: proceed.** R's kernel time (7.5-8.2 us, flat T=1..32 in profile 30027c28212946fd) is fixed latency, so a latency cut is reasonable, but 7.6 -> ~3.5 us is optimistic: per-warp top-8 keeps the same serial shuffle depth and removes ~14 of 16 block barriers. Realistic 1.5-3 us per T<=32 shape (+1-2% at decode_1/verify_1, likely within noise at decode_16/32). Proceed with no companions, with an old-vs-new and pdl 0/1 A/B inside prepare().

### Justification
R rows 7.6/7.8 (decode_1), 7.6/8.2 (decode_16), 7.5/7.9 (decode_32), 7.6/7.7 (verify_1) do not depend on T: a serial chain in one block (RMSNorm reduce, top-8 loop, threadfence + atomic ticket, last-block align with padding scans). (a) per-warp top-8 still ~80 dependent shuffles but 2 barriers instead of 16: ~0.5-1 us. (b) PDL between P and R is a real adjacency (_dl_forward launches P then R): firing launch_dependents after P's K loop hides R's launch gap and norm_w/bias loads, ~1-2 us, visible in the shape total, while R's own row grows by the griddepcontrol.wait time. (c) Binary search over padding slots helps only at bsm=16 (T=32), <0.5 us elsewhere. Incumbents: sliding/full decode_1 127.5/142.9, verify_1 190.8/192.4, decode_16 205.0/349.5, decode_32 245.8/529.7 (the hypothesis quotes stale 132.7/142.6). verify_16/32 are the in-run flat control. Correctness of (a): route_key unique per expert, so the global top-8 of per-warp top-8s gives the same ordered set; keep NaN handling identical.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (parent, master): (b) depends on dl['pdl']; read the pdl and PATH CHOSEN = DENSE-CHAIN lines. Shared TU: a compile error drops all of stage 1 to master.
- marlin-block-m-cap-16 (master): sets bsm, so the size (not sign) of (c) at decode_32.
- stack-align-par-sidestream-sliding1 (master): moves decode_1 denominators; no confound.
- post-expert-norm-triton-small-t, sglang-pipeline-parity-router-hilo-m8, fa3-hdim128-head-pairing-batched: not adjacent to P->R.

### Interference and A/B needs
No companion. Build requirements in prepare(): (1) keep master's R as dl_route_norm_align_v1 for timing only and print CUDA-event times for old and new R at T=1, 8, 32; (2) print the P->R pair captured in a CUDA graph replayed with pdl 0 and 1; (3) print R with and without the binary-search pad loop at T=32, or state bsm. No gate interaction with dense-chain-stage2-attn-side-pdl (keeps DL_MAX_TOKENS; stage 2 uses DL2_MAX_TOKENS); textual/build coupling only: both edit _DL_CUDA and _build_dense_chain; whichever lands second needs a 3-way restack keeping v2/-lineinfo and P's launch_dependents.

### Open questions
- Where do R's 7.6 us go? The in-prepare old-vs-new timing answers it.
- Does the judge's PDL probe choose pdl=1 here?
- Bigger cut: move the last-block align off the serial tail (per-block slots from a shared histogram)?
- If dense-chain stage 2 lands first, restack onto the v2 module; do not reuse DL2_MAX_TOKENS.

