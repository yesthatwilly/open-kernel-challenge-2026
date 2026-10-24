# muse_attention_decode · 2026-10-24T140710Z · check · sliding-fused-triton-decode-batched-deep-splits

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T140200Z-sliding-fused-triton-decode-batched-deep-splits.md` (branch `hyp/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits`).

- **Job:** `179dd013d3a347e4` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `f5cffd1`

## Goal
Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 1), so sliding decode at every batch (scored decode_16 and decode_32; tests batch 3 ctx 2047 page 64 and batch 4 ctx 6000 dist edges page 1) takes _sliding_forward, the one-launch _sliding_fused_kernel, instead of _pre_attention_kernel + prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + rms_norm_gate (five launches, 16.5-20.1 us GPU busy). (2) Change the split count for decode in _sliding_forward: S = 16 at every batch for q_len == 1 (each program owns ceil(2047/16) rounded to 64 = 128 old keys, two BLOCK_N=64 tiles), giving grid (B, 2, 16) = 512 CTAs at B=16 and 1024 at B=32, instead of _num_splits which gave 4 at B=16 and 2 at B=32 (8 and 16 serial tiles per program). Keep _num_splits itself unchanged for any other caller and keep B=1 at its current 16 (so decode_1 is byte-identical in behaviour). Everything else stays as master: the kernel body, BLOCK_N=64, num_warps=4, num_stages=2, the ticket-counter in-kernel combine (counters sized B*HKV from config, ws_acc/ws_ml sized B*HKV*S*M*D from config: 2 MB at B=32), the gate epilogue, and all full-layer and verify paths.

## What changed, and where
Worktree: `.worktrees/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits`, file `muse_attention_decode/submission.py` only (diff stat: 1 file, +5 / -4).

- `kernel_fn`: gate widened from `cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1` to `cfg["window"] and cfg["q_len"] == 1`, so sliding decode at every batch takes `_sliding_forward` (one `_sliding_fused_kernel` launch). Comment updated. Sliding verify (q_len 8) and every full layer (window == 0) still take the FA3 path below, unchanged.
- `_sliding_forward`: `splits = _num_splits(batch, hkv)` became `splits = 16 if length == 1 else _num_splits(batch, hkv)`, the form the review asked for so a 3-way merge with sliding-fused-triton-decode1-splits-32 stays clean. B=1 decode was already 16 (`_num_splits(1, 2) == 16`), so decode_1 behaves exactly as before. `_num_splits` itself is unchanged.
- Workspace and counters are still sized from config (`batch * hkv * splits`): ws_acc is 32*2*16*16*64*4 B = 2 MB at B=32, and counters are `B*HKV` int32 through the existing `_counters` cache keyed on (batch, hkv, device). Kernel body, BLOCK_N=64, num_warps=4, num_stages=2 and the ticket combine are untouched.
- No deviation from the spec. Review open question 3 does not apply yet: splits-32 has not landed on this base. If it lands first, the merge should keep the B=1 override from master and let this branch's `16` cover only batch > 1.
- Short windows (e.g. the test with batch 3, ctx 2047, page 64) leave the chunks for most s empty at S=16, because each chunk rounds up to 64. Those programs store m = -inf and l = 0, and the combine weights them 0. B=1 already runs this path with ctx 10. The kernel at B > 1 has run before (sliding-single-launch-fused-attention, at S=4 and S=2). S=16 at B > 1 has not run on GPU yet, so okc check is the first test of it.

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Evidence that split depth is the lever: sliding-single-launch-fused-attention ran this exact kernel with S from _num_splits and its sliding decode times tracked serial tiles per program, not bytes: decode_1 S=16 (2 tiles) 13.5 us, decode_16 S=4 (8 tiles) 23.9 us, decode_32 S=2 (16 tiles) 38.5 us, ~1.5-1.8 us per extra tile (a dependent page-table load then K/V gather each tile). Risks: the last-ticket program combines 16 partials of [16,64] fp32 from L2 (64 KB per (b,g)), measured fine at B=1; 1024 CTAs plus 64 counters is more atomics but one per program; if decode_32 is still behind, an ab-tune at S=32 (one tile per program) is the follow-up. Verify is deliberately excluded: the same kernel lost 2-2.6x on verify for a reason not explained by split depth (verify_1 at S=16 was 42 us). Disjoint from sliding-fused-triton-decode1-splits-32 (B=1 only) so the two can be merged by 3-way merge.

## Result (after the run)
- job `179dd013d3a347e4` (check), status done, passed True, tests 16/16
- speedup 1.3087x, 95% CI [1.2891, 1.3283], score 2877.8645440067467 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2930643047593426, "ci": [1.2736683401879525, 1.3211607975952195], "job": "295727fc68e14093"}; base result: muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-gate.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode-batched-deep-splits.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode-batched-deep-splits.patch`; branch `hyp/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits`
