---
problem: muse_decode_layer
slug: full-verify-cuda-splitkv-stage2
created: 2026-10-24T165224Z
base: master
area: FA3 call
change: >
  Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify
  blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16,
  296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build
  on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page
  64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched
  so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work
  items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query:
  master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as
  f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv
  heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query
  tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA
  merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert
  each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per
  warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t
  only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never
  dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @
  V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64]
  and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine:
  splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that
  request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16
  into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged
  lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else
  changes.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: "incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load"
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: stage 1 kernels unchanged; control
  - shapes: "*load=verify_1 *"
    direction: flat
    why: batch 1 outside the gate
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
risk: high
depends_on: full-decode-cuda-splitkv-port
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32
review: proceed
status: done
review_at: 2026-10-24T165634Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T003235Z
base_sha: f6cc1452343c4d93911ad7303c4a5d7878afd2fe
base_blob: b83f5ec1976c82d5b45dcd92b6c0818ce04eda9f
branch: hyp/muse_decode_layer/full-verify-cuda-splitkv-stage2
worktree: .worktrees/muse_decode_layer/full-verify-cuda-splitkv-stage2
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T004030Z
impl_commit: bed733be97f69ef7521118b8e47a44e5217070e9
submission_sha256: 56c1f88fd29a6cac1750a3f0f41a07be6fbde8e132bb0aa98a85a627d20fed20
norm_sha256: 6e636153b847e455107219497b47d258e56ae512d612318674a29737b9f16dd5
submission_blob: dd79676b5f7daeef54fdcbbc6044e5e09beae843
lint_warnings: 2
attempt_file: 2026-10-25T004030Z-check-full-verify-cuda-splitkv-stage2.md
prepared_at: 2026-10-25T004030Z
job: e9d5eeffddef4f70
dispatched_at: 2026-10-25T004037Z
outcome: regression
verdict_detail: regression
speedup: 1.1558993345373498
speedup_ci: [1.1385608445192896, 1.17323782455541]
harvested_at: 2026-10-25T013616Z
best_at_harvest: 1.2521147631671776
promotion: null
---

# full-verify-cuda-splitkv-stage2

## Change

Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16, 296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page 64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query: master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @ V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64] and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine: splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else changes.

## Expected

- `*layer=full load=verify_16 *` faster: incumbent paired FA3 153.3 us + combine + pad/unpad glue (~6.8 us x3) + norm-gate ~170 us of 604 us; the gather is the same as decode_16 where stage 1 kernel A ran ~111 us (attention), plus ~41 GFLOP of mma.sync that overlaps it: target <=130 us A+B, ~30-40 us off, +5-7% on the shape, ~+1.5-2% on the verify_16 load
- `*layer=full load=verify_32 *` faster: incumbent FA3 296.3 us + glue ~315 us of 933 us; target <=265 us A+B: ~40-50 us off, +4-6% on the shape, ~+1.5% on the load
- `*layer=full load=decode_*` flat: stage 1 kernels unchanged; control
- `*load=verify_1 *` flat: batch 1 outside the gate
- `*layer=sliding*` flat: untouched

## Risk

High: new causal-in-block masking and a 4x larger query tile; mma.sync compute (~41/82 GFLOP at verify_16/32) is no longer negligible against the gather, so if A exceeds ~150 us at verify_16 the fix is fp8 conversion once per stage in smem, not more splits. Run only after stage 1 lands or is judged on its replaced kernels. Not attempted in muse_attention_decode yet; a win here is portable back there.

## Milestones

Rewrite stage 2 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165634Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound stage-2 design. Warps own disjoint rows, the causal mask is right and the self-check is adequate, so it is worth one run as a rewrite. Its real risk is that A becomes ALU- or register-bound: four warps convert the same FP8 K/V tile four times, plus 4x mma.sync. Partials traffic is also 8x decode's, which stresses the <=130 us A+B target. Dispatch only after stage 1 (full-decode-cuda-splitkv-port) has run and its log shows the kernel compiled and self-checked, and on a check after the reset, not as a scored submit (risk high).

### Justification
Mechanism: at full verify_16/32 paired FA3 (153.3/296.3 us) streams the same page-1 KV gather as decode (146.8/289.0 us). FA3 verify costs only ~4% more than decode, so it is memory-bound, and LESSONS says this gather tops out at 1.3-1.7 TB/s. The attention port of the CUDA split-KV kernel A ran 111.4/245.5 us at decode_16/32 for the same bytes, so a verify kernel near that gather time would beat FA3 by ~30-45 us. The shape and load arithmetic checks out: verify_16 load is 46*462.4+16*603.9 = 30.9 ms, and 16*35 us = 0.56 ms is ~1.8%. verify_32: 16*45 of 44.3 ms is ~1.6%. The headline geomean moves ~0.5%, inside noise, so as the hypothesis says it must be judged on top_kernels per its Milestones. Decode rows (stage 1 kernel untouched), verify_1 (batch 1) and sliding rows are valid in-run controls. The causal bound t < cache_seqlens-7+i is correct (cache_seqlens = seq_lens+8, so token i sees keys <= seq_lens+i). Ragged lengths 1025/4097 already make chunks where tokens 0..6 are fully masked, which exercises the -inf partial path in the combine. Design soundness: the row-ownership layout (warp = kv head x 2 tokens x 8 heads, all 64 tokens of the tile) is the only one that keeps the accumulators in registers, and it uses all 16 mma rows (decode wasted 8). Weakest link 1, ALU: each tile's K and V are now converted e4m3->f16 by 4 warps per head instead of each warp converting a 16-token slice. That is 4x the cvt.rn.f16x2.e4m3x2 plus byte_perm work, about 4x2.5 MB of FP8 per SM at verify_16, on top of 4x the mma.sync (~41 GFLOP, ~70-100 us even at a good mma.sync rate). Compute that was hidden under the gather in decode may now be on the critical path. The stated fallback (an f16 smem tile per stage) costs ~32 KB per stage-slot and likely drops occupancy from 2 to 1 CTA/SM (stage 1 SMEM_A is ~93 KB, x2 = 186 of 228 KB). A better next-stage lever is QK^T with mma.sync m16n8k32 e4m3 (sm_89+/sm_90). Q and K are both already FP8, so it is exact, needs no K conversion and halves the QK mma count. Only V needs widening. Weakest link 2, registers: S 16x64 fp32 (32) + acc 16x64 (32) + Q A-frags for 4 k-steps (16) + P frags (16) + addressing is ~110-125 regs against the 128 cap of __launch_bounds__(256,2). Spills are likely. Weakest link 3, partials: 8 tokens x 16 heads x 64 fp32 = 32 KB per 1024-token item versus 4 KB in decode. That is ~40 MB written by A and re-read by the combine at verify_16 (~80 MB at verify_32), about 12% of the KV bytes, plausibly 10-20 us that the <=130 us A+B budget does not account for.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched: it defines the incumbent (paired FA3 + pad/unpad glue) that stage 2 replaces at verify_16/32. The saving is measured against the paired path, which is right, but it is conditional on pairing staying on master. If pairing were absent, the incumbent would be the unpaired FA3 (~7-13% slower on these shapes) and the gain would look larger. Not a confound that a companion needs to resolve.
- paired-heads-pad-unpad-triton (in the judge now, not landed): it shrinks exactly the pad/unpad glue (~6.8 us x3) that this hypothesis counts in its verify_16/32 saving. If it lands before stage 2's base is cut, the expected saving drops by ~10-20 us per shape. Re-baseline the incumbent kernel sum from whichever master the branch is cut from, and merge carefully because both edit the same paired branch of kernel_fn.
- marlin-block-m-cap-16: no mechanistic interaction (it is MoE, not attention). Marlin is still 48-51% of full verify_16/32 (284/501 us), and its run-to-run drift can mask a 30-45 us attention change in the per-shape row, so read the replaced kernels' own us in top_kernels.
- post-expert-norm-triton-small-t, sglang-pipeline-parity-router-hilo-m8: no interaction with the target shapes. The post-norm path is gated to small T and does not run at T=128/256. The router is unrelated.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage judged on the replaced kernels' own time against the incumbent and the floor. The decisive evidence is in the profile: splitkv_attn_verify_kernel + splitkv_combine_verify_kernel us at full verify_16/32, compared with paired FA3 main + combine + pad/unpad + FusedRMSNormGate from master. The run carries its own controls: decode rows run the unchanged stage 1 kernel, and verify_1, decode_1 and every sliding row run master code. An ab-revert of pairing would only re-measure FA3 unpaired, which is already known. The useful follow-ups (e4m3 mma for QK, a larger verify chunk to cut partials traffic, launch_bounds adjustments) belong in stage 3, not here. Sequencing note: stage 1 has not run on muse_decode_layer yet. Building on unharvested stage-1 code risks spending a check on a fallback run, so dispatch only after stage 1 harvests with '[splitkv] load_inline compiled' and no 'SELF-CHECK FAILED' in its log.

### Open questions
- What are the register count and spills of splitkv_attn_verify_kernel under __launch_bounds__(256,2)? Add -Xptxas -v (or print the cudaFuncGetAttributes numRegs/localSizeBytes) to the log so the harvest can tell ALU-bound from spill-bound.
- How does combine time scale with 8x larger partials (~40/80 MB at verify_16/32)? Would a 2048-token verify chunk (halving partials, still ~600 items vs 264 CTAs) be the cheaper stage-3 lever compared with conversion work?
- If A exceeds ~150 us at verify_16, is QK^T via mma.sync m16n8k32 e4m3 (no K conversion, exact) preferable to the f16 smem tile, which likely costs the second CTA per SM?
- Which incumbent glue is on the base master: does paired-heads-pad-unpad-triton land first? The expected verify savings should be restated against that base.
- Is the verify_16/32 'agentic' spread's max page_table width large enough that the workspace (B*max_chunks*8*16*64*4 bytes) inside graph capture is a concern? At verify_32 it is roughly 32*max_chunks*32 KB.


## Implementation notes

Edited only `.worktrees/muse_decode_layer/full-verify-cuda-splitkv-stage2/muse_decode_layer/submission.py` (+498/-30). **Base is stale** (claim reported `stale_base: true`): the worktree sits on stage 1's commit f6cc145, which predates master's current submission.py (stack-align-par-sidestream-sliding1, 1.2521). As instructed, stage 2 is built on stage 1 and NOT rebased; a merge to master will need a restack.

What changed:
- `splitkv_attn_verify_kernel` (new, stage 1 decode kernel byte-identical): same persistent 2xSM grid, 1024-token (b, chunk) items, smem page-index prefetch, 4-stage cp.async.cg ring (plain cp.async, no cache hints). Q for the item (8 tokens x 16 x 64 e4m3) is cp.async'd into a double-buffered smem tile with the indices and read into f16 A fragments at item start. Warp w = kv head w%2, query rows (w/2)*16..+16 (2 tokens x 8 heads) over the whole 64-token tile, so no intra-CTA merge. Per 64-token tile the warp runs two 32-token halves (S is 16 fp32 regs, to ease register pressure vs the 128-reg cap): S = QK^T with mma.sync m16n8k16 (K e4m3 -> f16 per warp in registers), causal mask `pos < cache_seqlens - 7 + i`, online exp2 softmax with -inf-safe m per row, P@V with mma.sync, fp32 acc 16x64 per warp. Masked/out-of-range tokens are zero-filled via cp.async src-size 0 and never dereferenced. Item end: each warp stores its two rows' fp32 acc as float4 and m, l to the workspace `[B*max_chunks, 8, 16, 64]` / `[B*max_chunks, 2, 8, 16]`.
- `splitkv_combine_verify_kernel`: grid (B*8, 16), one warp per (token, head): LSE merge (-inf partials weigh 0), x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024].
- `_splitkv_gate` admits q_len 8 once `_VERIFY_OK`; decode keeps q_len 1. `kernel_fn` dispatches q_len 1 to stage 1 and q_len 8 to `_full_verify_splitkv` (workspace from `page_table.shape[1]`, grid from SM count, no input values). Batch 1, sliding, page 64 unchanged.
- Self-check: the decode check is unchanged but now traced; a new verify check (lens 9000/1025/70/4097/1023, kv scale 1 and 0.25, device grid and 3 CTAs, plus a batch-33 case) against an FP32 causal reference (`_splitkv_reference` generalised over q_len). A verify-check failure disables only the verify path (`_VERIFY_OK=False`, master paired FA3 on verify, decode keeps stage 1); a decode-check failure still disables everything. Loud `PATH CHOSEN` lines per shape class.
- Build-brief items: `-lineinfo` added, module renamed `..._v2`, nvcc flags and TORCH_CUDA_ARCH_LIST printed (no wgmma/setmaxnreg/sm_90a-only PTX is used, so the arch list is left alone); every self-check launch prints before it and synchronizes plus prints after it (via a hook defined outside kernel_fn's call graph, so okc_lint sees no sync there); no spin waits exist (cp.async only), loops are bounded by B/n; no child process.
- Review fixes applied: the review is "proceed" with no required fix. Applied its guidance: print `cudaFuncGetAttributes` regs/localBytes/smem for the verify kernel to the log (open question 1). Not applied (stage 3 levers per the review): e4m3 m16n8k32 QK^T, larger verify chunk, f16 smem K tile. The review's sequencing advice (dispatch only after stage 1 harvests, on a check not a scored submit) is for the dispatcher; stage 1 has not been confirmed here.

Deviations: none from the spec except the 32-token sub-tile split inside the 64-token tile (same ring, same math) and Q fetched through smem alongside the indices rather than straight from global.

Uncertainties: register count / spills of the verify kernel under `__launch_bounds__(256,2)` (unmeasured, no GPU); the workspace is B*max_chunks*32 KB (about 100 MB at verify_32 with a wide agentic page table) allocated inside graph capture; the first launch cannot be exercised here.

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/synchronize/tensor prints in the kernel_fn path (the only synchronize calls are in prepare() and the self-check hook); workspace and grid come from shapes and device properties; the gate reads config only (checked in `git diff`).
- [x] Nothing derived from `data` is cached across calls: workspaces are fresh per call, `_EXT`/`_VERIFY_OK` are set only by the synthetic-data self-check.
- [x] Weights: n/a: no weight is touched or stored by this change.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; query FP8 E4M3 at most (unpaired cast as master; K/V widened to f16 exactly, fp32 accumulate, bf16 output).
- [x] Attention sees every key: all keys of every request in 1024-token chunks, causal inside the verify block; no skipping by weight (fully masked tokens are exactly the causal mask).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (grep for subprocess/multiprocessing/fork/spawn: none; load_inline is the same build as stage 1).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune, no host syncs in kernel_fn; first line `# okc: problem=muse_decode_layer` kept.

okc_lint output:

```
warn  host-sync    L1102: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1348: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed; `__pycache__` removed from the worktree.

## Pre-flight audit

Audited 2026-10-24 (model opus), by reading the worktree diff; no GPU here.

**Verdict: dispatch-with-fixes.** Rules OK.

Checked and found correct: the m16n8k16 A/B/C fragment layouts (Q rows A = token 2rg, rows B = token 2rg+1, the same k permutation for Q and K, the V byte_perm interleave, and the output store dims 16qi+nb / 16qi+8+nb); the causal mask `p < cache_seqlens - 7 + i`, where positions >= cache_seqlens are zero-filled through cp.async src-size 0 and never dereferenced, and fully masked rows give m=-inf, l=0, acc=0 and weigh 0 in the combine; that every (token, head) row of every chunk slot is written before the combine reads it; the WAR safety of the double-buffered idx/Q meta; SMEM_V of about 88.5 KB (x2 fits in 227 KB); score_scale and k_scale/v_scale; the gate (q_len 8 only after its own self-check, page 1, batch 2..64); and that each self-check is in prepare() only. The workspace is torch.empty from shapes only, per call, as in stage 1 (allowed, and capturable). It is about 87 MB at verify_32, since the judge sizes max_pages as ceil(80000*1.075)+8. That fits H200 memory easily. There is no ptxVersion gate, no added subprocess (beyond load_inline nvcc, which the rules allow), and nothing cached from inputs.

Fixes:
- `_splitkv_prepare`: a second prepare() call in the same process used to print "PATH CHOSEN: ... MASTER FA3 (no CUDA module)" while the CUDA path stayed live. That misleads the harvest. It now reports the path already chosen.

Residual risk: register pressure under __launch_bounds__(256,2) (about 100-120 live regs) may spill, which costs perf only. A device fault inside the verify self-check would poison the CUDA context and fail the run instead of falling back. Combine latency over about 85 chunks per warp is unmeasured.

## Result

- job `e9d5eeffddef4f70` (check), status done, passed True, tests 13/13
- speedup 1.1559x, 95% CI [1.1386, 1.1732], score 19569.43768728252 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-cuda-splitkv-port.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/full-verify-cuda-splitkv-stage2.result.json`, `muse_decode_layer/hypotheses/done/regression/full-verify-cuda-splitkv-stage2.patch`; branch `hyp/muse_decode_layer/full-verify-cuda-splitkv-stage2`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 144.97 | 146.13 | 0.9941 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 156.49 | 155.79 | 0.999 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 230.07 | 227.07 | 0.9963 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 329.13 | 327.75 | 0.9965 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 260.43 | 281.46 | 0.9961 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 490.19 | 509.44 | 0.9923 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 202.99 | 198.11 | 1.0007 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 220.46 | 216.07 | 1.0004 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 435.66 | 437.82 | 0.9887 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 590.55 | 682.67 | 0.8466 | lost |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 643.87 | 604.63 | 1.0087 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 909.53 | 1126.71 | 0.7973 | lost |

lost on full verify_16, verify_32. Expected hit: 3/5. Noise ±1.5%.

## Follow-ups

- rewrite stage 2 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer verify (q_len 8), batch 16/32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_decode_layer "<what to fix or tune>" --kind rewrite --depends-on full-verify-cuda-splitkv-stage2 --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.

