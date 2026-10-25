---
problem: muse_attention_decode
slug: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
created: 2026-10-25T001825Z
base: master
area: full-layer attention (verify, splitkv A8/B8)
change: >
  Stage 3 of the full-attention rewrite, built on the stage-2 code (branch
  hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebased onto
  current master; stage 2 regressed so it is not on master). Gate unchanged: _splitkv_verify_gate
  = full, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1. Keep stage 2's stream-K over
  64-token tiles, TP prefix, page-index prefetch, slot layout and masking math; replace the
  per-tile compute of A8 per REWRITE_READ_2026-10-25 section 2. New kernel
  splitkv_verify_wgmma_kernel in its OWN load_inline module muse_full_verify_wgmma_v1 (so a failed
  build cannot disable master's decode module), 256 threads = 2 warpgroups, __launch_bounds__(256,
  2), grid 2 x SM. WARPGROUP g OWNS KV HEAD g: its 64 rows (8 q heads x 8 block tokens) are
  exactly one wgmma M=64 tile. RING: all 256 threads still issue the 16-byte cp.async.cg gather
  for each 64-token tile (4-stage, 128 B K row + 128 B V row per token), but stores use the Hopper
  128B swizzle (16-byte chunk index XOR (row & 7), 1024-byte-aligned stage base), and completion
  is tracked by a per-stage FULL mbarrier (cp.async.mbarrier.arrive.noinc) and a per-stage EMPTY
  mbarrier both warpgroups arrive on when done; the thread issuing tile n+4 waits on EMPTY[stage].
  There is NO __syncthreads inside the tile loop. QK^T: wgmma.mma_async m64n64k32 f32.e4m3.e4m3, A
  = the request's fp8 Q block for head g from smem (64 rows x 64 dims, K-major, 64B swizzle,
  loaded once per segment), B = the K tile from the ring via descriptor with start offset 64*g
  bytes inside the 128-byte swizzled row (k-steps 2g, 2g+1 of the row, as CUTLASS advances
  descriptors within a swizzle atom); 2 instructions per tile; wgmma.fence / commit_group /
  wait_group 0 before reading S. Mask (stage 2 causal t <= seq_b - 8 + j, applied on any tile
  overlapping [seq_b - 8, seq_b), not just the last), online softmax exp2 domain per row from the
  wgmma accumulator layout (quad shuffles). PV in FP8 (as FA3): P rounded to e4m3 and
  byte-permuted (prmt) from the C-fragment layout to the wgmma register-A layout as FA3's fp8 path
  does; B = V^T: each warpgroup transposes ITS OWN head's 64x64 fp8 V half of the tile into a
  private 4 KB Vt smem buffer (tokens contiguous per dim, 128B swizzle; 32-bit loads + prmt +
  32-bit stores, double-buffered) synchronised by a 128-thread named barrier (bar.sync id, 128),
  never a CTA barrier; wgmma m64n64k32 with A from registers, B = Vt descriptor, O accumulated in
  fp32 (32 regs). This drops stage 2's cooperative f16 V-widen, its extra __syncthreads per tile,
  and each warp's full 48-mma tile. SHRINK B8: fold the combine + norm-gate into the kernel with a
  per-request ticket (atomicAdd after __threadfence; the last segment's CTA LSE-merges its
  request's slots, times v_scale, scaleless RMSNorm over 64 + sigmoid(gate), stores bf16 [B*8,
  1024], resets ticket[b] = 0; ticket is a module-level int32[64] zero buffer from import) so B8's
  launch and its (grid+B) x 8x16x64 fp32 re-read of all slots by a separate kernel go away; if the
  ticket path fails its self-check, keep stage 2's B8 and print that. BUILD/SAFETY (LESSONS
  2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a only (-gencode arch=compute_90a,code=sm_90a), all
  wgmma/mbarrier code under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets a
  device error flag and returns (no __trap), the flag read back after the canary (non-zero ->
  fallback), -lineinfo; print ptxas regs/smem/local bytes; first launch + self-check (stage 2's
  verify cases incl. grids 3/7/61, seq % 64 in 1..7, lengths < 8, kv_scale 1 and 0.25, two replays
  for the ticket reset) run IN-PROCESS (never in a child process: the rules statically reject
  submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25,
  corrected)): first confirm the sm_90a image actually loaded (log
  torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for
  splitkv_verify_wgmma_kernel; anything else turns the path OFF), then a tiny synthetic canary
  launch of splitkv_verify_wgmma_kernel (synthetic B=2, short lengths), each followed by
  torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any
  failure: full verify falls back to master (unpaired FA3 path), printing 'PATH CHOSEN: full
  verify master FA3 (fallback: <reason>)'; success prints 'PATH CHOSEN: full verify wgmma stream-K
  v1 (ticket|B8)'. Full decode (master splitkv v1) and all sliding shapes stay byte for byte. The
  first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky
  fault cannot be recovered in-process).
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "incumbent (master FA3 main+combine+prepare+norm-gate) 162.7 us, stage 2 A8+B8 241.7, floor 63.4. M1: 16/16 tests and < 241.7. M2 (read's gate): main kernel <= 128 us (1.15x stage-1 decode A) and replaced total < 162.7 (beats master). M3: <= 95 us (1.5x floor)"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "incumbent 301.3 us, stage 2 465.5, floor 126.8. M1: < 465.5. M2: main kernel <= 282 us and total < 301.3. M3: <= 190 us"
  - shapes: "*load=decode_* *layer=sliding* *layer=full load=verify_1 *"
    direction: flat
    why: outside _splitkv_verify_gate; master code
risk: high
depends_on: full-verify-cuda-streamk-splitkv
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8"
review: proceed
status: running
review_at: 2026-10-25T002202Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T002203Z
base_sha: ee7dd2035bcffc84a11faafb7b5574466ee3bcae
base_blob: 2b7e5dbf0fde0ae0163e5ecaefd87a5f0848e6b0
branch: hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
worktree: .worktrees/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
mode: null
restack_of: null
stale_base: true
---

# full-verify-wgmma-s3-warpgroup-per-kvhead-inproc

## Change

Stage 3 of the full-attention rewrite, built on the stage-2 code (branch hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebased onto current master; stage 2 regressed so it is not on master). Gate unchanged: _splitkv_verify_gate = full, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1. Keep stage 2's stream-K over 64-token tiles, TP prefix, page-index prefetch, slot layout and masking math; replace the per-tile compute of A8 per REWRITE_READ_2026-10-25 section 2. New kernel splitkv_verify_wgmma_kernel in its OWN load_inline module muse_full_verify_wgmma_v1 (so a failed build cannot disable master's decode module), 256 threads = 2 warpgroups, __launch_bounds__(256, 2), grid 2 x SM. WARPGROUP g OWNS KV HEAD g: its 64 rows (8 q heads x 8 block tokens) are exactly one wgmma M=64 tile. RING: all 256 threads still issue the 16-byte cp.async.cg gather for each 64-token tile (4-stage, 128 B K row + 128 B V row per token), but stores use the Hopper 128B swizzle (16-byte chunk index XOR (row & 7), 1024-byte-aligned stage base), and completion is tracked by a per-stage FULL mbarrier (cp.async.mbarrier.arrive.noinc) and a per-stage EMPTY mbarrier both warpgroups arrive on when done; the thread issuing tile n+4 waits on EMPTY[stage]. There is NO __syncthreads inside the tile loop. QK^T: wgmma.mma_async m64n64k32 f32.e4m3.e4m3, A = the request's fp8 Q block for head g from smem (64 rows x 64 dims, K-major, 64B swizzle, loaded once per segment), B = the K tile from the ring via descriptor with start offset 64*g bytes inside the 128-byte swizzled row (k-steps 2g, 2g+1 of the row, as CUTLASS advances descriptors within a swizzle atom); 2 instructions per tile; wgmma.fence / commit_group / wait_group 0 before reading S. Mask (stage 2 causal t <= seq_b - 8 + j, applied on any tile overlapping [seq_b - 8, seq_b), not just the last), online softmax exp2 domain per row from the wgmma accumulator layout (quad shuffles). PV in FP8 (as FA3): P rounded to e4m3 and byte-permuted (prmt) from the C-fragment layout to the wgmma register-A layout as FA3's fp8 path does; B = V^T: each warpgroup transposes ITS OWN head's 64x64 fp8 V half of the tile into a private 4 KB Vt smem buffer (tokens contiguous per dim, 128B swizzle; 32-bit loads + prmt + 32-bit stores, double-buffered) synchronised by a 128-thread named barrier (bar.sync id, 128), never a CTA barrier; wgmma m64n64k32 with A from registers, B = Vt descriptor, O accumulated in fp32 (32 regs). This drops stage 2's cooperative f16 V-widen, its extra __syncthreads per tile, and each warp's full 48-mma tile. SHRINK B8: fold the combine + norm-gate into the kernel with a per-request ticket (atomicAdd after __threadfence; the last segment's CTA LSE-merges its request's slots, times v_scale, scaleless RMSNorm over 64 + sigmoid(gate), stores bf16 [B*8, 1024], resets ticket[b] = 0; ticket is a module-level int32[64] zero buffer from import) so B8's launch and its (grid+B) x 8x16x64 fp32 re-read of all slots by a separate kernel go away; if the ticket path fails its self-check, keep stage 2's B8 and print that. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a only (-gencode arch=compute_90a,code=sm_90a), all wgmma/mbarrier code under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets a device error flag and returns (no __trap), the flag read back after the canary (non-zero -> fallback), -lineinfo; print ptxas regs/smem/local bytes; first launch + self-check (stage 2's verify cases incl. grids 3/7/61, seq % 64 in 1..7, lengths < 8, kv_scale 1 and 0.25, two replays for the ticket reset) run IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for splitkv_verify_wgmma_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of splitkv_verify_wgmma_kernel (synthetic B=2, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any failure: full verify falls back to master (unpaired FA3 path), printing 'PATH CHOSEN: full verify master FA3 (fallback: <reason>)'; success prints 'PATH CHOSEN: full verify wgmma stream-K v1 (ticket|B8)'. Full decode (master splitkv v1) and all sliding shapes stay byte for byte.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=full load=verify_16*` faster: incumbent (master FA3 main+combine+prepare+norm-gate) 162.7 us, stage 2 A8+B8 241.7, floor 63.4. M1: 16/16 tests and < 241.7. M2 (read's gate): main kernel <= 128 us (1.15x stage-1 decode A) and replaced total < 162.7 (beats master). M3: <= 95 us (1.5x floor)
- `*layer=full load=verify_32*` faster: incumbent 301.3 us, stage 2 465.5, floor 126.8. M1: < 465.5. M2: main kernel <= 282 us and total < 301.3. M3: <= 190 us
- `*load=decode_* *layer=sliding* *layer=full load=verify_1 *` flat: outside _splitkv_verify_gate; master code

## Risk

supersedes full-verify-wgmma-s3-warpgroup-per-kvhead, whose change requires a child-process self-check

## Milestones

Rewrite stage 3 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002202Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-3 design that targets the measured A8 bottleneck (2x stage-1's per-byte time, throughput-bound at the 128-reg cap) with a warpgroup-per-kv-head wgmma mapping that fits the M=64 shape exactly; proceed as an okc check, fixing the fallback to restore master's PAIRED FA3 path (not the unpaired one) and bounding every mbarrier wait so a protocol bug fails fast instead of hanging to the 15-minute timeout.

### Justification
A: The mechanism is plausible on H100. Each kv head's 8 q heads x 8 block tokens form exactly one wgmma M=64 tile. QK is 2 x m64n64k32 e4m3 per 64-token tile and PV is 2 more, so tensor-core work is negligible. Softmax is about 8K exp2 per 16 KB tile per CTA. At 2 CTA/SM that is roughly 40% of MUFU throughput at a ~3 TB/s gather rate, so the math can hide under the gather once the per-tile __syncthreads and cooperative f16 V-widen are gone. FA3 itself does this verify at ~2.0-2.2 TB/s (150.7 us main at verify_16). REWRITE_READ section 2 supports the diagnosis: A8 scales linearly with bytes (447.7/227.8 = 1.97x), at 1.45 TB/s against stage-1 decode A's 2.94 TB/s on the same gather machinery, at exactly 128 regs. That points to a per-tile throughput limit, not tail or latency. Register budget: S 32 + O 32 + P 8 + row stats + addresses comes to about 100 regs, under the 128 cap. Smem: stage 2's 90.6 KB + 16 KB Vt + 1024-B swizzle alignment comes to about 108 KB/CTA, which still allows 2 CTA/SM within 228 KB, but only just; print the occupancy. The expected directions match TASK.md: only full verify_16/32 (weight 16 each) are gated, every other shape keeps master, and the milestone numbers (incumbent 162.7/301.3, floors 63.4/126.8) match the read. The in-kernel ticket combine replaces B8 (13.9-17.8 us, about 2x FA3 combine+norm-gate) with a B-way parallel LSE merge of about grid/B slots x 32 KB per request at the tail. That costs an estimated 2-4 us and removes a launch, so it is a credible additional ~10 us. WEAKEST LINK: the cause of A8's 2x slowdown is inferred ('likely serializers'), not measured. Nothing was stamped per phase, and stage 2 sat at the register cap, so the wgmma restructure could hit a different limit: the fp8 C->A prmt, the per-warpgroup V transpose (32-bit loads + prmt + stores + named barrier on every tile), or mbarrier latency with only 4 stages. Even a perfect M2 (main kernel <= 1.15x stage-1 decode A) leaves verify at ~1.9-2.0x floor, so M3 (<= 1.5x floor, ~3.2 TB/s) needs a gather rate stage 1 never reached and is out of this stage's reach. That limits the stage, but it does not argue against running it. Secondary design risks: (1) the ring must store each stage's K rows contiguously at a 128-B pitch (separate K and V sub-buffers, SBO 1024 B per 8-row group) for the SW128 K-major B descriptor to be valid; if the stage-2 per-token layout interleaves K and V (256-B pitch), the descriptor and swizzle math must change. (2) Advancing the start address by 32*kstep + 64*g bytes inside a 128B-swizzle atom is legal (CUTLASS does it), but the Q A-operand uses a different swizzle (64B), so the two descriptors must be built independently.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv (stage 1, landed): the stage-3 kernel reuses its gather and page-prefetch machinery but builds in a separate module, so decode must stay byte-identical on master's muse_full_decode_splitkv_v1. Rebasing the stage-2 branch (which renamed and extended that module to v2 with A8/B8) must not carry v2's decode changes over. If B8 is kept as the ticket fallback, it has to be compiled into the new wgmma module, not into master's decode module. Full decode_32 drifts up to 6% between runs with identical kernels (LESSONS), so any headline movement there is noise; stage 2's '1.14x on decode_32' was already ruled a scoring artifact.
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup (+ ab-tune-batch-gate): master runs full verify_16/32 on the PAIRED FA3 path. The stage-3 gate turns pairing off for those shapes, so _pre_attention_kernel switches to the unpaired q layout there (a small, real time change at 2-4 us, included in the shape's time but not in the 'replaced' set). More importantly, the spec's fallback says 'master (unpaired FA3 path)'. Master is the paired path. An unpaired-FA3 fallback would be slower than master (pairing gained ~1.17x), so a fallback run would read as a regression attributable to nothing in the kernel. The fallback must clear the verify gate so that paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv and not splitkv_verify evaluates True again.
- fused-preattention-triton: it supplies the FP8 q that becomes the wgmma A operand (unpaired layout [B*8, 16, 64]). The Q-block smem load must match that layout exactly; it does not change the measurement otherwise.
- sliding-deep-splits-on-cuda-splitkv / sliding-fused-triton-decode1-gate: different shapes, no interaction beyond import-time build cost (a third load_inline module adds ~45 s+ of the 15-minute budget).

### Interference and A/B needs
No companions. This is a rewrite stage judged on the replaced kernels' own profile time (splitkv_verify_wgmma_kernel, plus B8 if the ticket falls back) against the incumbent's 162.7/301.3 us and floors of 63.4/126.8 us. That reading does not depend on the headline, so neither a gain nor a loss is ambiguous in a way an extra run would resolve. An ab-revert of the pairing hypotheses would only change the fallback path, which is not what is being measured. A control isolating the ticket combine from the wgmma compute is unnecessary: the profile shows whether B8 is present, and the log line 'PATH CHOSEN ... (ticket|B8)' names the variant. The essential attribution guard is procedural: before reading any number, grep the log for the sm_90a build line, binaryVersion/ptxVersion, the canary/self-check lines and PATH CHOSEN, and confirm splitkv_verify_wgmma_kernel appears in the verify_16/32 profile (LESSONS 2026-10-25: wgmma builds have silently measured the fallback). The next stage is the follow-up.

### Open questions
- Does the rebased stage-2 ring lay out K and V rows interleaved per token (256-B pitch) or as separate 64x128-B K and V sub-tiles? The SW128 K-major B descriptor needs the latter.
- Does ptxas confirm <= 128 regs with 0 local bytes, and does cudaOccupancyMaxActiveBlocksPerMultiprocessor report 2 at ~108 KB smem? If it reports 1, the grid of 2 x SM runs in two waves and the result is not a test of the design.
- How large is the in-kernel ticket merge tail (last CTA per request merges ~grid/B slots x 32 KB)? A clock64 stamp in the self-check would separate it from the main loop in the next stage.
- If M2 is missed with no spills, can the next stage get per-phase clock64 timings (gather wait, QK, softmax, P permute, V transpose, PV) in the self-check, so stage 4 is not built on another inferred bottleneck?
- Should the fallback be master's paired FA3 path (correct) rather than the 'unpaired FA3 path' the spec names? Confirm the implementer gates pairing on the new module's success flag.

