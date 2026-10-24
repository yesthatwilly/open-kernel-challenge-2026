---
problem: muse_attention_decode
slug: sliding-single-launch-fused-attention
created: 2026-10-24T081233Z
base: master
area: sliding-layer attention (whole-path rewrite)
change: >
  Stage 1 of a rewrite of the sliding-layer path. Built on
  hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding
  layers (config window != 0) kernel_fn launches ONE new Triton kernel, _sliding_fused_kernel, in
  place of _pre_attention_kernel + flash_attn_with_kvcache (its prepare_varlen_num_blocks, main
  and FlashAttnFwdCombine launches) + rms_norm_gate, and returns its bf16 [B, L, Hq*64] output.
  Full layers (window == 0) keep the base's path unchanged (_pre_attention_kernel + FA3 with
  scheduler_metadata + rms_norm_gate). GRID (B, Hkv, S): one program per (request b, kv head g,
  window split s); each program serves the group's 8 query heads for all L block tokens as one M =
  8*L row tile (rows (j, h); decode M = 8 padded to 16 for tl.dot, verify M = 64). S is a
  constexpr chosen from config batch so the launch has ~64-128 CTAs: S = 16 at B=1 (128 keys per
  split), 4 at B=16, 2 at B=32 (cap 16, power of two; S = 1 allowed). RANGE: seq = seq_lens[b];
  old keys are positions [lo, seq-1], lo = max(0, seq - (window-1)) (the union of the L tokens'
  windows; token j at p_j = seq + j attends t in [p_j - window + 1, p_j]); the old range is cut
  into S contiguous chunks of ceil(count/S) rounded up to BLOCK_N (64); a chunk may be empty
  (short-context tests ctx 10 / 2047), which still writes m = -inf, l = 0 and takes a ticket.
  PROLOGUE (every program): load the 8*L query rows from qkv (head h*64), scaleless RMSNorm (eps
  1e-5, mean of 64 squares), interleaved RoPE with pos = positions[b*L+j] and the cached inv_freq
  exactly as in the base kernel, keep fp32 in registers and feed tl.dot as bf16 (finer than E4M3;
  or E4M3 x E4M3 tl.dot as FA3 does; either is allowed and must be decided once, recorded in the
  attempt). Only program s == S-1 also builds the L new key/value rows of head g: RMSNorm+RoPE the
  k row, divide k and v by k_scale/v_scale, cast to E4M3, store them at cache_loc[b*L+j] (the
  entries the judge checks), and keep the E4M3 values in registers; it attends these dequantized
  E4M3 registers (not the fp32 values, so the output matches the entries stored) as one extra
  16-row key block masked to i < L and causal i <= j. No program ever reads a new row back from
  the cache, so there is no store-then-load race; the old rows [lo, seq-1] it reads are never
  written in this call. MAIN LOOP: for each BLOCK_N block of its chunk, rows = page_table[b, t //
  PAGE] * PAGE + t % PAGE (PAGE = config page_size constexpr: 1 scored, 64 in tests, same code),
  load k_cache.view(-1, Hkv, 64)[rows, g] and v likewise as [BLOCK_N, 64] E4M3 (64 contiguous
  bytes per row), num_stages 2-3 so the gathers are in flight, convert to bf16 (exact), S = q @
  k^T in fp32 times sm_scale*qk_scale_factor*k_scale, mask per row j: t >= p_j - window + 1 (t <=
  p_j holds for all old keys), online softmax in fp32 (exp2 with log2e folded), acc += P(bf16) @
  V, v_scale applied once at the end. Every key in each token's window is scored: no skipping.
  COMBINE (in-kernel, S > 1): each program writes (m, l, acc[M, 64]) fp32 to a workspace [B*Hkv,
  S, M, 66], then tl.debug_barrier and t = tl.atomic_add(counter[b*Hkv+g], 1, sem=acq_rel); the
  program that draws t == S-1 reads all S partials (cache_modifier .cg), rescales by exp(m_s -
  m_max), sums, divides by l; then resets counter to 0 for the next replay. The workspace is
  torch.empty per call (graph pool; shape from config only); the int32 counters are a
  zero-initialised buffer allocated once per (B, Hkv, device) before capture and left zeroed by
  every call, holding no input-derived data. With S == 1 the program goes straight to the
  epilogue. EPILOGUE (the last program): per (j, h) row o = acc/l; r = rsqrt(mean(o^2) + 1e-5)
  over 64; load the gate logits qkv[b*L+j, (Hq+2*Hkv)*64 + h*64 : +64], o * r * sigmoid(gate) in
  fp32, cast bf16, store to out[b, j, h*64 : (h+1)*64] (no intermediate bf16 rounding of o: finer
  than SGLang). kv_scale != 1 tests: k_scale folded into the score scale, v_scale into o before
  the norm, divisions applied on the new rows' store. No host syncs, no .item, no value-dependent
  branching or allocation; everything branches only on config (batch, q_len, page_size, window).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: incumbent 16.77 us (SGLang 20.90, floor 0.11); replaced kernels 15.1 us GPU in 5 launches (FA3 8.2, combine 2.5, prepare 1.8, pre-attn 1.5, norm-gate 1.0); 32 CTAs x 128 keys x 128 B = 512 KB in one launch, target <= 6 us replaced time
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: incumbent 20.80 us (SGLang 27.01, floor 1.65); replaced 17.9 us; 128 CTAs stream 8 MB, target <= 7 us
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: incumbent 25.14 us (SGLang 31.41, floor 3.29); replaced 21.2 us; 128 CTAs stream 16 MB, target <= 9 us
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: incumbent 17.12 us (SGLang 22.83, floor 0.12); replaced 16.2 us; M=64 rows per program, same 512 KB, target <= 7 us
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent 22.50 us (SGLang 29.34, floor 1.86); replaced 20.4 us, target <= 8 us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: incumbent 28.72 us (SGLang 34.30, floor 3.72); replaced 26.7 us, target <= 10 us
  - shapes: "*layer=full*"
    direction: flat
    why: full layers keep the base path (pre-attn kernel + FA3 + rms_norm_gate) byte for byte; control
risk: high
depends_on: fused-preattention-triton
priority: 1
kind: rewrite
controls_for: null
replaces: "sliding-layer attention after the QKV projection: _pre_attention_kernel + FA3 (+combine/prepare) + rms_norm_gate"
review: proceed
status: done
review_at: 2026-10-24T083713Z
review_companions: null
claimed_at: 2026-10-24T083713Z
base_sha: 48b24f974a04559fb27967b595d013641507059c
base_blob: f14211b4b0e88f8b3f683d9f9275b76a64b66e02
branch: hyp/muse_attention_decode/sliding-single-launch-fused-attention
worktree: .worktrees/muse_attention_decode/sliding-single-launch-fused-attention
mode: check
implemented_at: 2026-10-24T084220Z
impl_commit: db5fa4d87590c050d3108322d81e5b226fdd427d
submission_sha256: dcba0f814ecd82f4b07cf9ebdb1cf209a498adf98e6d5fe8685aca4247e9aa5d
norm_sha256: f612c1ff67ac7b11c8d0882fbf0d6144cee4b75972c597aeea83ef4f7bb9202d
submission_blob: ddc819f35e8b6aa78cd2ad84e3be00a879437e8e
lint_warnings: 0
attempt_file: 2026-10-24T084220Z-check-sliding-single-launch-fused-attention.md
prepared_at: 2026-10-24T084220Z
job: f389af196ddf47b1
dispatched_at: 2026-10-24T084230Z
stage: harvested
rewrite_stage: 1
outcome: regression
verdict_detail: regression
speedup: 0.9452410112427269
speedup_ci: [0.9310623960740859, 0.9594196264113677]
harvested_at: 2026-10-24T103710Z
best_at_harvest: 1.1284429079326423
promotion: null
---

# sliding-single-launch-fused-attention

## Change

Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (config window != 0) kernel_fn launches ONE new Triton kernel, _sliding_fused_kernel, in place of _pre_attention_kernel + flash_attn_with_kvcache (its prepare_varlen_num_blocks, main and FlashAttnFwdCombine launches) + rms_norm_gate, and returns its bf16 [B, L, Hq*64] output. Full layers (window == 0) keep the base's path unchanged (_pre_attention_kernel + FA3 with scheduler_metadata + rms_norm_gate). GRID (B, Hkv, S): one program per (request b, kv head g, window split s); each program serves the group's 8 query heads for all L block tokens as one M = 8*L row tile (rows (j, h); decode M = 8 padded to 16 for tl.dot, verify M = 64). S is a constexpr chosen from config batch so the launch has ~64-128 CTAs: S = 16 at B=1 (128 keys per split), 4 at B=16, 2 at B=32 (cap 16, power of two; S = 1 allowed). RANGE: seq = seq_lens[b]; old keys are positions [lo, seq-1], lo = max(0, seq - (window-1)) (the union of the L tokens' windows; token j at p_j = seq + j attends t in [p_j - window + 1, p_j]); the old range is cut into S contiguous chunks of ceil(count/S) rounded up to BLOCK_N (64); a chunk may be empty (short-context tests ctx 10 / 2047), which still writes m = -inf, l = 0 and takes a ticket. PROLOGUE (every program): load the 8*L query rows from qkv (head h*64), scaleless RMSNorm (eps 1e-5, mean of 64 squares), interleaved RoPE with pos = positions[b*L+j] and the cached inv_freq exactly as in the base kernel, keep fp32 in registers and feed tl.dot as bf16 (finer than E4M3; or E4M3 x E4M3 tl.dot as FA3 does; either is allowed and must be decided once, recorded in the attempt). Only program s == S-1 also builds the L new key/value rows of head g: RMSNorm+RoPE the k row, divide k and v by k_scale/v_scale, cast to E4M3, store them at cache_loc[b*L+j] (the entries the judge checks), and keep the E4M3 values in registers; it attends these dequantized E4M3 registers (not the fp32 values, so the output matches the entries stored) as one extra 16-row key block masked to i < L and causal i <= j. No program ever reads a new row back from the cache, so there is no store-then-load race; the old rows [lo, seq-1] it reads are never written in this call. MAIN LOOP: for each BLOCK_N block of its chunk, rows = page_table[b, t // PAGE] * PAGE + t % PAGE (PAGE = config page_size constexpr: 1 scored, 64 in tests, same code), load k_cache.view(-1, Hkv, 64)[rows, g] and v likewise as [BLOCK_N, 64] E4M3 (64 contiguous bytes per row), num_stages 2-3 so the gathers are in flight, convert to bf16 (exact), S = q @ k^T in fp32 times sm_scale*qk_scale_factor*k_scale, mask per row j: t >= p_j - window + 1 (t <= p_j holds for all old keys), online softmax in fp32 (exp2 with log2e folded), acc += P(bf16) @ V, v_scale applied once at the end. Every key in each token's window is scored: no skipping. COMBINE (in-kernel, S > 1): each program writes (m, l, acc[M, 64]) fp32 to a workspace [B*Hkv, S, M, 66], then tl.debug_barrier and t = tl.atomic_add(counter[b*Hkv+g], 1, sem=acq_rel); the program that draws t == S-1 reads all S partials (cache_modifier .cg), rescales by exp(m_s - m_max), sums, divides by l; then resets counter to 0 for the next replay. The workspace is torch.empty per call (graph pool; shape from config only); the int32 counters are a zero-initialised buffer allocated once per (B, Hkv, device) before capture and left zeroed by every call, holding no input-derived data. With S == 1 the program goes straight to the epilogue. EPILOGUE (the last program): per (j, h) row o = acc/l; r = rsqrt(mean(o^2) + 1e-5) over 64; load the gate logits qkv[b*L+j, (Hq+2*Hkv)*64 + h*64 : +64], o * r * sigmoid(gate) in fp32, cast bf16, store to out[b, j, h*64 : (h+1)*64] (no intermediate bf16 rounding of o: finer than SGLang). kv_scale != 1 tests: k_scale folded into the score scale, v_scale into o before the norm, divisions applied on the new rows' store. No host syncs, no .item, no value-dependent branching or allocation; everything branches only on config (batch, q_len, page_size, window).

## Expected

- `*layer=sliding load=decode_1 *` faster: incumbent 16.77 us (SGLang 20.90, floor 0.11); replaced kernels 15.1 us GPU in 5 launches (FA3 8.2, combine 2.5, prepare 1.8, pre-attn 1.5, norm-gate 1.0); 32 CTAs x 128 keys x 128 B = 512 KB in one launch, target <= 6 us replaced time
- `*layer=sliding load=decode_16 *` faster: incumbent 20.80 us (SGLang 27.01, floor 1.65); replaced 17.9 us; 128 CTAs stream 8 MB, target <= 7 us
- `*layer=sliding load=decode_32 *` faster: incumbent 25.14 us (SGLang 31.41, floor 3.29); replaced 21.2 us; 128 CTAs stream 16 MB, target <= 9 us
- `*layer=sliding load=verify_1 *` faster: incumbent 17.12 us (SGLang 22.83, floor 0.12); replaced 16.2 us; M=64 rows per program, same 512 KB, target <= 7 us
- `*layer=sliding load=verify_16 *` faster: incumbent 22.50 us (SGLang 29.34, floor 1.86); replaced 20.4 us, target <= 8 us
- `*layer=sliding load=verify_32 *` faster: incumbent 28.72 us (SGLang 34.30, floor 3.72); replaced 26.7 us, target <= 10 us
- `*layer=full*` flat: full layers keep the base path (pre-attn kernel + FA3 + rms_norm_gate) byte for byte; control

## Risk

Milestones (judged on the replaced kernels' GPU time from the profile, not the headline): M1 all 16 tests pass and every sliding shape is faster than the incumbent; M2 each sliding shape's single kernel <= 50% of the replaced 15-27 us; M3 within 3x the floor + ~3 us launch on B=16/32. Subsumes triton-window-attention-sliding, fused-sliding-preattn-into-window-kernel and fused-sliding-normgate-epilogue (the three-step stack done at once) and makes fa3-single-split-sliding / fa3-window-trimmed-page-table moot for sliding layers if it lands; churnsubs/reviewer may park those. Risks: (1) verify q_len=8 causal inside the block: the new-key block mask is i <= j and the old-key mask is t >= seq + j - window + 1 per row; the edges test plants keys at p0-2047 (inside for j=0, outside for j>=1) and p0-2048 (outside) and must pass. (2) page_size 64 tests (ctx 10 and ctx 2047 sliding run at page 64): the row formula is generic; if it fails, branch on config page_size != 1 to the incumbent path (allowed: config branch) and say so, scored shapes are page 1. (3) numerics vs FA3: fp32 online softmax and an S-way LSE combine differ from FA3 only in reduction order; bf16 queries are finer than FA3's E4M3; the output bound is 2.5x the E4M3-query error + 1.5%. Attend the E4M3-rounded new k/v, not fp32. Near-zero-row rule: eps inside sqrt over the mean of 64 squares, as reference._rms. (4) the ticket combine: partial stores must be visible before the atomic (acq_rel, gpu scope) and the last program must read with .cg; the counter must be reset to 0 by that program or the next replay deadlocks/mis-combines; if Triton memory ordering proves flaky, fall back to S = 1 (no combine) and accept the B=1 bandwidth cost, recorded as a stage-1 result. Persistent zeroed counters are workspace, not input-derived data, but the reviewer should confirm that reading of the caching rule. (5) register pressure: M=64 x 64 fp32 acc plus a 64x64 score tile at 4 warps; BLOCK_N 32 if Triton spills; decode pads M 8 -> 16. (6) single-program gather throughput at page 1 (64 B rows): bytes in flight per CTA = BLOCK_N x 128 B x num_stages; if B=1 stays above ~6 us, raise S rather than BLOCK_N. Stage 2 candidates: E4M3 tl.dot for QK^T, warp-specialised loads, a persistent grid.

## Milestones

Rewrite stage 1 of `sliding-layer attention after the QKV projection: _pre_attention_kernel + FA3 (+combine/prepare) + rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T083713Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite that goes after the part of sliding-layer time that is launches and latency. The profile shows 15-27 us of replaced GPU time spread over 5 launches to move 0.5-16.8 MB. The only landed hypothesis is absorbed by this change on sliding layers and left byte-identical on full layers, so one run is interpretable from the single kernel's profile line with full layers as the drift control. No companions are needed.

### Justification
Mechanism: on sliding shapes FA3 main (7.7-14.9 us), FlashAttnFwdCombine (2.5-4.3 us), prepare_varlen_num_blocks (1.8 us), _pre_attention_kernel (1.5-4.1 us) and rms_norm_gate (~1 us) cost 15-27 us. The bytes moved are 512 KB at B=1 (2048 keys x 2 kv heads x 128 B K+V) and 8.4/16.8 MB at B=16/32, against floors of 0.1/1.6/3.3 us. So the time is dependent launches and DRAM latency, not bandwidth. LESSONS records the one place fusion paid: launch-dominated 17-36 us attention_decode shapes (1.127x). The expected directions fit the weights. Sliding shapes carry weight 46 against 16 for full, so sliding is ~63% of load time at B=1 (46*16.8 vs 16*27.9) and ~16-24% at B=16/32. The large geomean lever is therefore decode_1/verify_1. The full-layer 'flat' row is correct: that path is byte-identical. Window algebra checks out against FA3 window_size (window-1, 0). lo = seq-(window-1) is the j=0 lower bound and so the union over the block. The old-key mask t >= p_j-window+1 and the new-key mask i <= j reproduce causal-in-block, and the dist=edges keys at p0-2047 (in for j=0, out for j>=1) and p0-2048 (out) fall on the right side. Attending the E4M3-rounded new k/v from registers matches the judge, which attends your own new entries, and avoids a store-then-load race. Feeding bf16 queries is finer than the E4M3 allowance. Weakest link: the B=1/verify_1 critical path is a dependent chain, and the per-shape targets are optimistic. The chain is cold qkv + page_table load -> row-index gather of K/V -> partial store -> gpu-scope atomic ticket -> winner re-reads S partials -> gate load -> store, inside one launch. That is realistically ~5-8 us, not <=6 us. At B=32 LESSONS measured the page-size-1 row gather at 1.3-1.7 TB/s, which puts 16.8 MB at ~10-13 us, not <=9. Both still clear the incumbent's 15-27 us replaced time by a wide margin, so milestone 2 (beat the incumbent) is very likely, but 'within 1.5x the floor' (milestone 3) is unreachable at B=1 (floor 0.1 us) for any design. Judge stage 1 on replaced time vs incumbent, not on the floor ratio. The second risk is the Triton ticket combine's memory ordering: release by one thread after tl.debug_barrier. Make it robust before trusting a pass on its own; see open_questions.

### Confounders on the current kernel
- fused-preattention-triton (landed, master base): not a confounder for the sliding rows. This change removes _pre_attention_kernel from the sliding path and re-does its work (q/k RMSNorm+RoPE, k/v E4M3 store at cache_loc) inside the new kernel, so a sliding gain or loss belongs to the new kernel and not to an interaction. Its only residual effect is that the incumbent being compared against is 1.5-4.1 us cheaper than SGLang's 5-launch pre-attention. That makes the measured gain a lower bound of the gain against SGLang, not an inflated one. On full layers it stays byte-for-byte, which makes the full rows a same-run drift control.

### Interference and A/B needs
No companion is warranted. (i) An ab-revert of fused-preattention-triton is meaningless on sliding layers because the change subsumes that kernel there, and it would only perturb the full-layer control. (ii) There is no landed knob to tune that the new kernel depends on. (iii) A control isolating a sub-change, such as S=1 with no in-kernel combine, or keeping rms_norm_gate separate, is not needed to read this result. The profile gives the single kernel's own time per sliding shape against the 15-27 us replaced sum, the full-layer rows bound run-to-run drift (~1.5-2.5%), and the expected effect (several us on 17-29 us shapes) is far above noise. Both a gain and a loss are attributable to the new kernel. If B=1 disappoints, the S=1 fallback and a raised S are next-stage material, not companions, as the rewrite rules prescribe.

### Open questions
- Ticket-combine ordering in Triton: a scalar tl.atomic_add(sem='acq_rel') is issued by one thread after tl.debug_barrier. Is that cumulative over the other threads' partial stores on sm_90 as Triton lowers it? The conservative CUDA pattern is a gpu-scope fence by every thread, then the barrier, then the atomic, then an acquire on the winner side. Validate it with a stress loop of many replays at B=1 and S=16 locally before the run. A sporadic mis-combine would surface as a flaky correctness failure on a timed call, not on the tests.
- Online-softmax -inf guard: rows can be fully masked inside a block, for example verify rows j>=1 at the window's lower edge or tail blocks past seq-1. m_new must stay -inf-safe (exp2(-inf - -inf) = NaN), and the combine must use m_max from finite partials. The S-1 partial is always finite because it holds the new keys, but the winner may be any program. Check this explicitly. The 'No NaN or Inf' rule fails the whole run.
- Does the S-1 program's extra prologue (building, storing and attending L new k/v rows) lengthen the critical path at B=1, given that it also runs the main loop over a full chunk? Consider giving chunk S-1 fewer old keys or none, so its new-key work overlaps the others' gathers.
- Milestone 3 (within ~1.5x of the floor) cannot be met at B=1/verify_1, where the floor is 0.1 us after the judge's ~4 us fixed-cost subtraction. Churnsubs should judge stage 1 on M1/M2 and on whether B=16/32 approach the page-1 gather ceiling (~1.3-1.7 TB/s per LESSONS), and not treat a missed M3 at B=1 as a failure.
- If this lands, the queued sliding-only FA3 hypotheses become moot for sliding layers: triton-window-attention-sliding, fused-sliding-preattn-into-window-kernel, fused-sliding-normgate-epilogue, fa3-single-split-sliding, fa3-window-trimmed-page-table and fa3-num-splits-sliding. Churnsubs should park them rather than run them against the new base.


## Implementation notes

Implemented 2026-10-24 in the worktree's `muse_attention_decode/submission.py` only (+236 lines, nothing removed). Uncommitted, and not run: this node has no GPU, torch or Triton.

**Dispatch.** `kernel_fn` returns `_sliding_forward(data)` when `config["window"] != 0`. Otherwise it falls through to master's code, which is unchanged byte for byte: `_pre_attention_kernel` + FA3 with scheduler_metadata + `rms_norm_gate`. Full layers are the drift control.

**Kernel `_sliding_fused_kernel`.** Grid `(B, Hkv, S)` with `num_warps=4` and `num_stages=2`. It uses these helpers: `_norm_rope`, `_online_update`, `_gate_epilogue` and `_gpu_fence`.
- **S per batch**, from `_num_splits` (config only). S is the power of two that keeps `B*Hkv*S <= 128`, capped at 16. That gives B=1 → 16 (32 CTAs, 128 keys per split), B=16 → 4 (128 CTAs), B=32 → 2 (128 CTAs). The tests get B=3/4 → 16 and B=5 → 8.
- **Row tile.** M = max(16, next_pow2(GROUP*L)): 16 at decode (8 real rows plus 8 padding rows) and 64 at verify. Row r maps to token j = min(r // GROUP, L-1) and head g*GROUP + r % GROUP. Padding rows load q = 0 and stay finite, and their stores are masked.
- **Prologue.** RMSNorm and interleaved RoPE run on full [R, 64] tiles. A second load at `d ^ 1` gives the pair-swapped row: y = x·cos + sign·x_swap·sin, with sign −1 on even d and +1 on odd d. That is the same arithmetic as the base kernel's even/odd form, with `inv_freq[d//2]` and `pos = positions[b*L+j]`.
- **Query precision for QK^T: bf16 operands.** q is normalized, rotated, multiplied by qk_scale_factor and only then cast to bf16, which is exactly where the reference rounds. Each K/V block is converted E4M3 → bf16, which is exact. The score scale is `sm_scale*log2e*k_scale` and accumulation is fp32. P is fed to the P@V dot as bf16 with fp32 accumulation, and v_scale is applied once at the end.
- **New rows, program s == S-1 only.** It does RMSNorm+RoPE on k, divides k and v by k_scale and v_scale, casts to E4M3, and stores at `cache_loc[b*L+i]` (masked i < L). It then attends the dequantized **E4M3 register values** as one 16-row key block masked to `i < L & i <= j`. No new row is ever read back from the cache. The caches are passed as `uint8` views and loads and stores bitcast to `float8e4nv`, which avoids any fp8 `other=` cast on masked loads.
- **Old keys** are positions `[lo, seq-1]` with `lo = max(0, seq-2047)`. They are split into S contiguous chunks of `ceil(count/S)` rounded up to 64, and empty chunks are allowed (seq 10 / 2047). Each BLOCK_N = 64 block is fetched through the page table: `row = page_table[b, t//PAGE]*PAGE + t%PAGE`, and PAGE is the config page_size constexpr. Pages of 1 and of 64 run the same code, so there is no fallback to master's path. Each row is masked `t >= seq + j - 2047`; `t <= p_j` always holds for old keys.
- **Every key in every token's window is scored.** Edge check: lo = p0-2047, so p0-2048 is never loaded. For j=0, p0-2047 is in the window. For j >= 1 the per-row mask excludes it. Within the block, i <= j holds for q_len = 8. Nothing is skipped based on a score.
- **Online softmax is -inf safe.** `m_use = where(m_new == -inf, 0, m_new)`. Then `alpha = exp2(m_i - m_use)` and `p = exp2(qk - m_use)`, so a row with no valid key keeps m = -inf, l = 0 and acc = 0, and exp2(-inf - -inf) never occurs. In the combine, m_max is the maximum over the partials; partial S-1 is finite on every row because its diagonal i = j is always valid, padding rows included. As a second guard, m_max is also clamped with `where(-inf → 0)`. Each partial's weight is `where(m_s == -inf, 0, exp2(m_s - m_max))`. With S == 1, l > 0 on every row because that program holds the new keys.
- **Epilogue** (the S==1 program, or the combine winner). o = acc/l·v_scale in fp32, with no bf16 rounding of o. Then `rsqrt(mean(o²)+1e-5)` over 64 is multiplied by sigmoid(gate) loaded from `qkv[tok, (Hq+2Hkv)*64 + h*64 :]`, the result is cast to bf16, and it is stored to `out[b, j, h*64:]`.

**How ordering is ensured (ticket combine, S > 1).**
1. Each thread stores its partials: acc goes to `ws_acc [B*Hkv*S, M, 64]` and m, l go to `ws_ml [B*Hkv*S, 2, M]`, both fp32 and `torch.empty` per call. This layout is two buffers in place of the spec's single [.., M, 66] buffer, a layout change only.
2. Every thread executes `fence.acq_rel.gpu` through `tl.inline_asm_elementwise(..., is_pure=False)` on an [M] tensor. With 128 threads and M ≤ 64 the elements are replicated, so all threads execute it.
3. `tl.debug_barrier()`.
4. ONE scalar `tl.atomic_add(cnt_ptr + b*Hkv + g, 1, sem="acq_rel", scope="gpu")`.
5. `tl.debug_barrier()`.
6. The program that draws `ticket == S-1` fences again on every thread (acquire side). It then reloads all S partials with `cache_modifier=".cg"`, combines them, runs the epilogue, and stores 0 to its counter.

The counters are int32 `[B*Hkv]`, allocated with `torch.zeros` once per (B, Hkv, device) in `_COUNTERS` on the first eager call. They hold no input-derived data and are left at 0 after every launch. On S == 1 they are not allocated or touched; dummy pointers are passed instead. Everything runs on the current stream, with no side stream.

**Deviations and uncertainties.**
- **No warm-up in `prepare()`.** This problem has no `prepare` hook: the technical rules say muse_attention_decode is the one problem without weights or prepare. The Triton variants compile on the first eager call per config, as the base's `_pre_attention_kernel` already does.
- **Unverified Triton features.** `tl.inline_asm_elementwise` (the fence) and `scope=` on `atomic_add` both need Triton ≥ 3.0-ish. If either fails to compile, the first fix is to drop the fence: the barrier plus the acq_rel atomic remain.
- **The S-1 program does the new-row work and also a full chunk.** The review's suggestion of a shorter chunk for S-1 was not done; it is stage-2 material.
- **Untested.** The stress test (many replays at B=1, S=16) could not be run here. A sporadic mis-combine would show up as a flaky correctness failure on a timed call.
- **Pipelining of the gathers** relies on Triton's pipeliner with `num_stages=2` on a loop with runtime bounds and indirect loads.

**Readout plan.** Judge on the single `_sliding_fused_kernel` time per sliding shape in the profile, against the incumbent's replaced 15-27 µs. The replaced kernels are `_pre_attention_kernel`, FA3 main, FlashAttnFwdCombine, prepare_varlen_num_blocks and rms_norm_gate. Rough targets, adjusted per the review: about 5-8 µs at decode_1 and verify_1, and about 10-13 µs at B=32. The full-layer rows must stay flat (drift ±1.5-2.5%). Milestone 1: all 16 tests pass. Milestone 2: each sliding shape is faster than the incumbent's replaced time. Milestone 3 (≤ 1.5× floor) cannot be reached at B=1 and verify_1, where the floor is 0.1 µs. At B=16/32, compare instead against the page-1 gather ceiling of about 1.3-1.7 TB/s. A missed milestone 3 at B=1 is not a failure.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-single-launch-fused-attention diff` (+236 lines in submission.py only):
- [x] CUDA-graph capturable. There is no `.item()`, `.cpu()`, `synchronize()` or tensor print. `torch.empty` allocations take their shapes from config. S, M, LN, PAGE, WINDOW and ROPE are all derived from config. The only runtime branches are on program ids and on the ticket value inside the kernel; there is no host control flow on input values. The counters are zero-initialised on the first eager call, not during capture.
- [x] Nothing derived from `data` is cached across calls. `_COUNTERS` is keyed by (B, Hkv, device) and holds zeros, restored to 0 by every launch. `_inv_freq` is the base's theta-only cache.
- [x] n/a: weights. This problem has no weights and no `prepare()`.
- [x] Activations BF16 where SGLang keeps BF16. The KV cache stays FP8 E4M3: new rows are stored as E4M3, and old and new keys are attended as E4M3 values. Queries are bf16, which is finer than the E4M3 allowance. The output is bf16.
- [x] Attention sees every key. Every key in `[max(0, p_j-2047), p_j]` is scored for each token j: old keys in S contiguous chunks covering `[lo, seq-1]` exactly, plus the new keys with i <= j. There is no top-k, sparsity, block skipping or weight-based skipping.
- [x] No timing patches, no access to harness frames, I/O or secrets, no module edits, no reference import, no subprocesses and no network. The imports are unchanged.
- [x] All work happens inside the timed call. No output is cached or replayed; the workspace is per call and the counters are pure synchronisation state.
- [x] Only the named area changed: the sliding path in `kernel_fn` plus the new helpers. The full-layer path is byte-identical. The first line, `# okc: problem=muse_attention_decode`, is kept.
- [x] No `triton.autotune`, no side stream, and no host syncs in kernel_fn.

Lint (`okc_lint.py ... --problem muse_attention_decode`):
```
CLEAN: .worktrees/muse_attention_decode/sliding-single-launch-fused-attention/muse_attention_decode/submission.py
```
`python3 -I -m py_compile` passed, and the `__pycache__` it left in the worktree was deleted.

## Result

- job `f389af196ddf47b1` (check), status done, passed True, tests 16/16
- speedup 0.9452x, 95% CI [0.9311, 0.9594], score 3984.486448644851 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.1284429079326423, "ci": [1.1115162643136527, 1.1453695515516318], "job": "c297440e53334f34"}; base result: muse_attention_decode/hypotheses/done/gain/fused-preattention-triton.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.patch`; branch `hyp/muse_attention_decode/sliding-single-launch-fused-attention`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 16.77 | 13.51 | 1.2371 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.87 | 27.44 | 1.0164 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.8 | 23.87 | 0.8669 | lost |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 194.45 | 195.49 | 1.0006 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 25.14 | 38.47 | 0.6496 | lost |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 377.92 | 377.67 | 1.0023 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.12 | 42.34 | 0.4055 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.02 | 29.79 | 0.9823 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.5 | 47.23 | 0.4761 | lost |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 198.32 | 198.62 | 0.997 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.72 | 60.84 | 0.473 | lost |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 383.11 | 383.98 | 0.9996 | flat |

gained on sliding decode_1; lost on sliding decode_16, decode_32, verify_1, verify_16, verify_32. Expected hit: 2/7. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `sliding-layer attention after the QKV projection: _pre_attention_kernel + FA3 (+combine/prepare) + rms_norm_gate`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on sliding-single-launch-fused-attention --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.

