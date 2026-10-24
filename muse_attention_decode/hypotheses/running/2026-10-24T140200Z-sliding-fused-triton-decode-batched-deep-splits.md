---
problem: muse_attention_decode
slug: sliding-fused-triton-decode-batched-deep-splits
created: 2026-10-24T140200Z
base: master
area: sliding-layer attention (batched decode, split depth)
change: >
  Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from
  (window and batch == 1 and q_len == 1) to (window and q_len == 1), so sliding decode at every
  batch (scored decode_16 and decode_32; tests batch 3 ctx 2047 page 64 and batch 4 ctx 6000 dist
  edges page 1) takes _sliding_forward, the one-launch _sliding_fused_kernel, instead of
  _pre_attention_kernel + prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine +
  rms_norm_gate (five launches, 16.5-20.1 us GPU busy). (2) Change the split count for decode in
  _sliding_forward: S = 16 at every batch for q_len == 1 (each program owns ceil(2047/16) rounded
  to 64 = 128 old keys, two BLOCK_N=64 tiles), giving grid (B, 2, 16) = 512 CTAs at B=16 and 1024
  at B=32, instead of _num_splits which gave 4 at B=16 and 2 at B=32 (8 and 16 serial tiles per
  program). Keep _num_splits itself unchanged for any other caller and keep B=1 at its current 16
  (so decode_1 is byte-identical in behaviour). Everything else stays as master: the kernel body,
  BLOCK_N=64, num_warps=4, num_stages=2, the ticket-counter in-kernel combine (counters sized
  B*HKV from config, ws_acc/ws_ml sized B*HKV*S*M*D from config: 2 MB at B=32), the gate epilogue,
  and all full-layer and verify paths.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "incumbent 20.66 us (FA3 9.9 + combine 3.2 + prepare 1.9 + pre-attn 1.7 + norm 1.2); the same kernel at S=4 measured 23.87 us (8 serial dependent tiles per program) while at B=1 with S=16 (2 tiles) it is 13.1 us shape / 10.5 us kernel: per-program serial tiles, not bytes (8 MB, 1.7 us floor), set the time, so 2 tiles at 512 CTAs should land ~12-14 us; 46 weight, ~1.12x on the decode_16 load"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: incumbent 24.89 us; S=2 (16 tiles) measured 38.47 us; S=16 at 1024 CTAs (all resident, ~8/SM at 4 warps) ~13-16 us; ~1.08x on the decode_32 load
  - shapes: "*layer=sliding load=decode_1*"
    direction: flat
    why: same gate and S=16 as master
  - shapes: "*layer=full*"
    direction: flat
    why: untouched control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 keeps FA3
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T140502Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T140502Z
base_sha: 4c53fa864c3e6f5d9693120cc5890d9a0bf15753
base_blob: 387dcb6a0e6137b74f7f65b048dce5a3fcbe2008
branch: hyp/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits
worktree: .worktrees/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits
mode: check
implemented_at: 2026-10-24T140709Z
impl_commit: f5cffd11a6757b3d8e7e38d257cf290306303e71
submission_sha256: d285e4b08c7517d338c4120e9811874eaabde4fbbfcb426518ed57250fbf4708
norm_sha256: 8e2acb87640ea56d475c4a9e51dd61618999a773a4dc4736c8ffdbcaff1ed8a8
submission_blob: 9baa6cb5dafe3330ff14eac219f5b5a0fb90f127
lint_warnings: 0
attempt_file: 2026-10-24T140710Z-check-sliding-fused-triton-decode-batched-deep-splits.md
prepared_at: 2026-10-24T140710Z
job: 179dd013d3a347e4
dispatched_at: 2026-10-24T140720Z
---

# sliding-fused-triton-decode-batched-deep-splits

## Change

Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 1), so sliding decode at every batch (scored decode_16 and decode_32; tests batch 3 ctx 2047 page 64 and batch 4 ctx 6000 dist edges page 1) takes _sliding_forward, the one-launch _sliding_fused_kernel, instead of _pre_attention_kernel + prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + rms_norm_gate (five launches, 16.5-20.1 us GPU busy). (2) Change the split count for decode in _sliding_forward: S = 16 at every batch for q_len == 1 (each program owns ceil(2047/16) rounded to 64 = 128 old keys, two BLOCK_N=64 tiles), giving grid (B, 2, 16) = 512 CTAs at B=16 and 1024 at B=32, instead of _num_splits which gave 4 at B=16 and 2 at B=32 (8 and 16 serial tiles per program). Keep _num_splits itself unchanged for any other caller and keep B=1 at its current 16 (so decode_1 is byte-identical in behaviour). Everything else stays as master: the kernel body, BLOCK_N=64, num_warps=4, num_stages=2, the ticket-counter in-kernel combine (counters sized B*HKV from config, ws_acc/ws_ml sized B*HKV*S*M*D from config: 2 MB at B=32), the gate epilogue, and all full-layer and verify paths.

## Expected

- `*layer=sliding load=decode_16*` faster: incumbent 20.66 us (FA3 9.9 + combine 3.2 + prepare 1.9 + pre-attn 1.7 + norm 1.2); the same kernel at S=4 measured 23.87 us (8 serial dependent tiles per program) while at B=1 with S=16 (2 tiles) it is 13.1 us shape / 10.5 us kernel: per-program serial tiles, not bytes (8 MB, 1.7 us floor), set the time, so 2 tiles at 512 CTAs should land ~12-14 us; 46 weight, ~1.12x on the decode_16 load
- `*layer=sliding load=decode_32*` faster: incumbent 24.89 us; S=2 (16 tiles) measured 38.47 us; S=16 at 1024 CTAs (all resident, ~8/SM at 4 warps) ~13-16 us; ~1.08x on the decode_32 load
- `*layer=sliding load=decode_1*` flat: same gate and S=16 as master
- `*layer=full*` flat: untouched control
- `*load=verify_*` flat: q_len 8 keeps FA3

## Risk

Evidence that split depth is the lever: sliding-single-launch-fused-attention ran this exact kernel with S from _num_splits and its sliding decode times tracked serial tiles per program, not bytes: decode_1 S=16 (2 tiles) 13.5 us, decode_16 S=4 (8 tiles) 23.9 us, decode_32 S=2 (16 tiles) 38.5 us, ~1.5-1.8 us per extra tile (a dependent page-table load then K/V gather each tile). Risks: the last-ticket program combines 16 partials of [16,64] fp32 from L2 (64 KB per (b,g)), measured fine at B=1; 1024 CTAs plus 64 counters is more atomics but one per program; if decode_32 is still behind, an ab-tune at S=32 (one tile per program) is the follow-up. Verify is deliberately excluded: the same kernel lost 2-2.6x on verify for a reason not explained by split depth (verify_1 at S=16 was 42 us). Disjoint from sliding-fused-triton-decode1-splits-32 (B=1 only) so the two can be merged by 3-way merge.

## Adversarial review

Reviewed 2026-10-24T140502Z by an adversarial subagent (model opus).

**Verdict: proceed.** This one is worth running and needs no companions. The gate-widened kernel at _num_splits depth has already been measured (sliding-single-launch-fused-attention: 23.87 / 38.47 us at S=4 / S=2), so a single S=16 run on decode_16 and decode_32 shows what split depth does on its own, and the per-shape result can be read whether it wins or loses.

### Justification
The mechanism is plausible. At sliding decode with M=16 query rows (G=8, L=1) and page_size 1, each program's old-key loop is a chain of dependent loads that miss a cold L2: a page-table load, then a gather of 64-byte K/V rows, for every 64-key tile. Each extra serial tile costs about one DRAM latency pair. The decode_16 and decode_32 bytes (8 / 16 MB, floors 1.6 / 3.3 us) are far below the measured times, so the time is latency and concurrency, not bandwidth. There are three data points: B=1/S=16/2 tiles at 13.5 us, B=16/S=4/8 tiles at 23.9 us and B=32/S=2/16 tiles at 38.5 us. They fit about 1.5-1.8 us per extra tile. Going to 512 or 1024 CTAs with 2 tiles each raises the bytes in flight per SM about 4-8x, which is what a latency-bound gather needs. The expected numbers are consistent with TASK.md weights. Taking sliding decode_16 from 20.66 to ~13 us saves ~350 us on a 3461 us load (~1.11x). Taking decode_32 from 24.89 to ~14.5 us saves ~480 us on a 5921 us load (~1.08-1.09x). That gives ~1.03x geomean, which should clear the +/-1.5% CI and best's CI high (1.321) only narrowly. decode_1 is byte-identical (_num_splits(1,2) already returns 16) and verify stays on FA3, so the flat predictions are sound. Weakest link: in the three data points, tiles per program, grid size and total bytes all change together with B. The data therefore does not show that the ~10.5 us fixed cost at B=1 (32 CTAs) stays fixed at 512-1024 CTAs. At B=32 the grid is 1024 CTAs of 4 warps, and with num_stages=2 dot tiles (~16-32 KB smem each) that is about 7 resident per SM on 132 SMs. A partial second wave, or 64 combines each reading 16 x 4 KB partials, could push decode_32 to 16-18 us, still a win but a smaller one. A secondary concern is that verify_1 at S=16 (2 tiles, M=64) took 42 us. Something other than tile count clearly costs a lot in this kernel, possibly the M-tile size. That does not apply to M=16 decode, but it is a reminder that the tile-count model does not cover everything.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate: this is the code being widened, not a confounder. B=1 keeps S=16 and its path is byte-identical, so the decode_1 result stays as landed. A merge with the queued sliding-fused-triton-decode1-splits-32 (B=1 only) would touch the same split-count line in _sliding_forward. Keep the edit to a `16 if length == 1 else _num_splits(...)` form so a 3-way merge stays clean.
- fused-preattention-triton: the incumbent's sliding decode_16/32 time includes its 1.7-1.9 us _pre_attention_kernel launch. Folding it into the fused kernel is part of the intended gain, not an interaction. Its full-layer and verify use is unchanged.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / paired-heads-launch-cleanup-ab-tune-batch-gate: full layers only (window == 0). The sliding path never reads the paired layout, so there is no shared launch sequence, memory traffic or numerical path.

### Interference and A/B needs
No companion is warranted. The control that isolates sub-change (1), the gate widened with _num_splits depth, was already run as sliding-single-launch-fused-attention. That run's sliding path matches this kernel body verbatim, and the full-layer changes since then do not touch sliding shapes. So the S=16 effect can be read straight against those numbers: 23.87 -> ? at decode_16 and 38.47 -> ? at decode_32. The FA3 incumbent is in the same run as the reference. Reverting a landed hypothesis would change nothing on the sliding shapes. Results come per shape, so a loss on decode_32 alone can still be read. If decode_32 lags, the follow-up is a stacked ab-tune (S=32, one tile per program, or S=8 to fit one wave), not a companion now. Attention has only 2 checks left today, which also argues against spending one on a companion.

### Open questions
- At B=32, how many CTAs per SM can be resident (smem with num_stages=2 on a 2-iteration loop, plus registers)? If it is below 8, 1024 CTAs is about 1.1 waves. num_stages=1 or S=8 might fit one wave better. Read the occupancy from the profile if it is available.
- Is the ~10.5 us kernel time at B=1 (32 CTAs, 2 tiles) mostly the serialized combine tail (last ticket reloads 16 partials from L2 after a gpu-scope fence)? If so, it stays fixed at large B and caps decode_16/32 near 11-13 us. The S=32 follow-up would double the combine work and may not help.
- If sliding-fused-triton-decode1-splits-32 lands first, check that this branch's split-count edit still leaves B=1 at whatever S master then uses (i.e. it should only override for batch > 1).


## Implementation notes

Worktree: `.worktrees/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits`, file `muse_attention_decode/submission.py` only (diff stat: 1 file, +5 / -4).

- `kernel_fn`: gate widened from `cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1` to `cfg["window"] and cfg["q_len"] == 1`, so sliding decode at every batch takes `_sliding_forward` (one `_sliding_fused_kernel` launch). Comment updated. Sliding verify (q_len 8) and every full layer (window == 0) still take the FA3 path below, unchanged.
- `_sliding_forward`: `splits = _num_splits(batch, hkv)` became `splits = 16 if length == 1 else _num_splits(batch, hkv)`, the form the review asked for so a 3-way merge with sliding-fused-triton-decode1-splits-32 stays clean. B=1 decode was already 16 (`_num_splits(1, 2) == 16`), so decode_1 behaves exactly as before. `_num_splits` itself is unchanged.
- Workspace and counters are still sized from config (`batch * hkv * splits`): ws_acc is 32*2*16*16*64*4 B = 2 MB at B=32, and counters are `B*HKV` int32 through the existing `_counters` cache keyed on (batch, hkv, device). Kernel body, BLOCK_N=64, num_warps=4, num_stages=2 and the ticket combine are untouched.
- No deviation from the spec. Review open question 3 does not apply yet: splits-32 has not landed on this base. If it lands first, the merge should keep the B=1 override from master and let this branch's `16` cover only batch > 1.
- Short windows (e.g. the test with batch 3, ctx 2047, page 64) leave the chunks for most s empty at S=16, because each chunk rounds up to 64. Those programs store m = -inf and l = 0, and the combine weights them 0. B=1 already runs this path with ctx 10. The kernel at B > 1 has run before (sliding-single-launch-fused-attention, at S=4 and S=2). S=16 at B > 1 has not run on GPU yet, so okc check is the first test of it.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits diff` (1 file, +5 / -4).

- [x] CUDA-graph capturable: the diff adds no `.item()`, `.cpu()`, sync or tensor print. Both edited predicates branch only on `config` (`window`, `q_len`, with `length = cfg["q_len"]`). Allocations depend on config shapes only.
- [x] Nothing derived from `data` is cached across calls. The only cache is the existing `_COUNTERS`, keyed on config (batch, hkv, device) and holding no data. The diff adds no new state.
- [x] n/a: weights. This problem has no weights or `prepare()`, and the diff touches none.
- [x] Activations: the kernel body is unchanged. BF16 q/K/V dot operands, FP8 E4M3 cache stores and fp32 accumulation, as before. The split count does not change precision.
- [x] Attention sees every key: the chunking covers [lo, seq - 1] across S splits, plus the new keys in split S - 1, for any S. No skipping is added.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The diff is two config expressions and comments.
- [x] All work happens inside the timed call. No output caching or replay.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. No `triton.autotune` added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`, rc 0):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK (rc 0). Removed `__pycache__` from the worktree.
