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
stage: implemented
claimed_at: 2026-10-25T010823Z
base_sha: 9d09de0cc3ef315d6ef3aab918058321b698da3f
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/dl-route-warp-topk-pdl
worktree: .worktrees/muse_decode_layer/dl-route-warp-topk-pdl
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T011431Z
impl_commit: 91e7a28f9d1bf45c1ddf0a814defd1063e119621
submission_sha256: 5819b26d5299dc4d105a4a32a9136c7f40e03d54b23d5321a56a4aeb0c1c4144
norm_sha256: ef1a8d1fbe8dafaf1cc746358fc23e1ff311698753ea211e78c54ce0ea3e75d8
submission_blob: 1dc2735e85cedd0e25f03663f989c4330ac8d50a
lint_warnings: 33
attempt_file: 2026-10-25T011431Z-check-dl-route-warp-topk-pdl.md
prepared_at: 2026-10-25T011431Z
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


## Implementation notes

All edits in the worktree's muse_decode_layer/submission.py (`_DL_CUDA`, `_build_dense_chain`, `_dl_route*`, `_dl_self_check`, `_dl_setup`). Not compiled or run (no GPU/nvcc on this node).

- Kernel P: `griddep_launch()` after the main K loop of every CTA, before the split-K partial store. Always executed; harmless without the attribute.
- Kernel R: master's kernel kept verbatim as `dl_route_norm_align_v1_kernel` (variant 2, prepare() timing and comparison only; kernel_fn never launches it). New `dl_route_norm_align_kernel<BSEARCH>`: per-warp top-8 (lane-local taken flag, lane r keeps round-r winner in s_cand[warp*8+r]), one barrier, warp 0 takes the top-8 of 64 candidates (2 per lane), one barrier. route_key, score, tie order, weights, topk_ids/topk_w/local_ids writes, NaN handling (same keys, same `taken ? 0 : key` rule) unchanged.
- Review fix 4: before `griddepcontrol.wait` only norm_w (16 floats/thread as 4 float4) and bias[e] are read; logits, ei and local_start come after it.
- PDL launch of R via `cudaLaunchKernelEx` + programmatic serialization when `pdl` (a prepare() decision, `dl["pdl"]`); the wrapper takes `pdl` and `variant` (0 new/binary-search pad, 1 new/linear pad, 2 v1; v1 is never launched with the attribute).
- Binary-search pad (5 steps over s_seg0) for padding slots and block_experts; the linear scan stays as BSEARCH=false for the A/B.
- Review fix 1-3 (prepare() prints, CUDA-event, 20 launches per graph): `R T=1/8/32 bsm=.. old(v1) / new / new-linear-pad`, and `P->R pair T=1/8/32 new pdl=0 / new pdl=1 / old-R pdl=0` (10 pairs per graph). bsm is printed.
- Review fix 6: the self-check adds a bit-exact comparison of both new variants against v1 (xn, topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded) at T=1/8/32; a mismatch fails the self-check, so the run falls to master's path.
- Review fix 7: `-lineinfo` added; self-check prints before and sync+print after P, R (via `_dl_forward_traced`, self-check only), R alone and each R variant. Module name unchanged. No sm_90a-only PTX beyond griddepcontrol (existing -arch=sm_90a), so no TORCH_CUDA_ARCH_LIST change.
- Review fix 5: DL_MAX_TOKENS and every bound constant untouched. The review wins over the spec on: keeping the old R for timing, the P->R graph A/B, and norm_w/bias-only pre-wait reads (same as the spec's intent). No new mbarrier/spin waits.
- Deviation: trace syncs live in `_dl_forward_traced` (not `_dl_forward`) because okc_lint flagged syncs in a function reachable from kernel_fn.

## Rules check

- [x] CUDA-graph capturable: kernel_fn path is `_dl_forward` (P launch + R launch, no sync/.item/.cpu); verified in `git diff` that syncs only appear in prepare()-only functions.
- [x] Nothing derived from `data` is cached across calls (only prepare constants norm_w/bias/ws/cnt).
- [x] Weights: no change to weight repacking.
- [x] Activations: BF16 xn as before, same arithmetic as v1; KV/query paths untouched.
- [x] Attention: n/a: not touched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (no subprocess/fork anywhere in the diff).
- [x] All work inside the timed call; no output caching or replay (v1 R and timing run only in prepare()).
- [x] Gate: DL_MAX_TOKENS unchanged; build/self-check failure falls back to master, never raises.

okc_lint: `OK with warnings` (only prepare()-scope host-sync warnings); py_compile ok; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the code and emulating it in Python. **Verdict: dispatch-with-fixes.**

- Top-8: a Python emulation of v1 against the per-warp + warp-0 merge found 0 mismatches over 3000 cases (random scores, many exact ties, +/-NaN). route_key, the order of the weight sum, norm_w indexing (float4 [8 tid, 8 tid + 8) and 2048 + that) and the score and bias match v1, so the bit-exact self-check can pass.
- PDL: only norm_w and bias are read before griddepcontrol.wait. local_start, logits, ei and cnt are read and every write happens after it. P triggers after its K loop and has no early return. R's wait covers P's completion and memory flush, and P waits on its own predecessor, so the chain holds transitively.
- Binary-search pad: emulation at bsm 1/8/16/64 with empty segments and the all-no-local case matched the linear scan for every slot and block_experts (0 mismatches).
- Gates and rules: DL_MAX_TOKENS is unchanged. There is no ptxVersion gate. kernel_fn calls `_dl_forward` (no syncs; the traced variant is reached only from prepare()). Build failure, self-check failure and setup exceptions all land on MASTER with a PATH CHOSEN line, and _dl_time_r is wrapped in try.
- Fix 1: replaced `auto fn = variant == 0 ? launch_r<true> : launch_r<false>;` with an explicit if/else. A conditional over template-ids may be rejected by nvcc's front end, and a compile error would drop all of stage 1 to master.
- Fix 2: added a 16-byte alignment TORCH_CHECK on norm_w, which the new R reads as float4. A misaligned pointer now raises (self-check fails, so the run goes to master) instead of a sticky misaligned-address fault.
- okc_lint: OK with warnings (prepare()-only syncs). py_compile: ok. __pycache__: removed.

Residual risk: the code was never compiled, so an nvcc error in the shared TU would put all of stage 1 on master. The eager self-check syncs between P and R, so only the graph-replay case exercises the real PDL overlap.
