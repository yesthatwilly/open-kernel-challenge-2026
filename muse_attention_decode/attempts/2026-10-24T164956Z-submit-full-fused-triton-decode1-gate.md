# muse_attention_decode · 2026-10-24T164956Z · submit · full-fused-triton-decode1-gate

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T140225Z-full-fused-triton-decode1-gate.md` (branch `hyp/muse_attention_decode/full-fused-triton-decode1-gate`).

- **Job:** `af3c6a46badf4e50` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `96908ab`

## Goal
Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forward, replacing the unpaired path there (_pre_attention_kernel + prepare_varlen_num_blocks + FA3 main 14.8 us + FlashAttnFwdCombine 7.2 us + rms_norm_gate). Kernel edits, all behind new constexprs so the sliding instantiation compiles to the same code: (a) FULL: tl.constexpr (= window == 0): lo = 0 instead of max(seq - (WINDOW - 1), 0), and the old-key validity is tmask only (every old key t < seq <= p_row is attended; no window lower edge); ROPE is already off for full layers (rope_theta 0) and the new key/value is built, stored at cache_loc and attended from registers exactly as on sliding; (b) the split combine loops: keep tl.static_range(S) when S <= 16 (sliding, unchanged) and use a runtime range(S) loop when S > 16 (a constexpr branch), so a deep split does not unroll 64 copies. _sliding_forward chooses, for FULL only, S = 64 and BLOCK_N = 128 (80000 old keys -> 1250 per split, ~10 tiles of 128), grid (1, 2, 64) = 128 CTAs (about one per SM), num_warps=4, num_stages=3; workspace ws_acc [128, 16, 64] fp32 (512 KB) and ws_ml from config; the counters come from _counters(1, 2) as today. Sliding keeps S from _num_splits, BLOCK_N=64, num_stages=2. scheduler_metadata is not read on this path. Full batch>1 (paired FA3), full verify_1 and every sliding shape are unchanged.

## What changed, and where
Worktree `.worktrees/muse_attention_decode/full-fused-triton-decode1-gate`, file `muse_attention_decode/submission.py` only (uncommitted; +130/-33).

- `_sliding_fused_kernel`: new constexpr `FULL`. When FULL, `lo = 0` (static branch) and the old-key validity is `tmask[None, :]` only; the sliding branch keeps `max(seq - (WINDOW - 1), 0)` and the per-row lower edge verbatim. New key/value build, cache store and in-register attend are untouched (ROPE is off because rope_theta is 0).
- Combine, deviation from the md per the review's REQUIRED (2): no runtime `range(S)` loop. A static `if S > 16` branch loads all S (m, l) partials as one `[S, M]` tile (S=64, M=16 -> 4 KB each), computes m_max, the weights and l_sum vectorized, then streams acc as 8 unrolled, independent `[CS=8, M, D]` 3D loads (32 KB each) reduced over the split axis with `tl.sum(axis=0)`. `S <= 16` keeps the original `tl.static_range(S)` code (re-indented only), so sliding instantiations (FULL=False, S<=16, BLOCK_N=64, num_stages=2) compile the same. `CS: tl.constexpr = 8` declared at the kernel top (compile-time only).
- `_sliding_forward`: `full = window == 0` -> S = `_FULL_SPLITS` (64), BLOCK_N=128, num_stages=3; else S = `_num_splits`, BLOCK_N=64, num_stages=2. num_warps=4 both. Workspace ws_acc [128, 16, 64] fp32 / ws_ml [128, 2, 16] and `_counters(1, 2)` as for sliding. scheduler_metadata not read.
- New `_full_fused_gate(cfg)`: `_FULL_FUSED and window == 0 and batch == 1 and q_len == 1 and tp8 (16/2) and head_dim 64 and page_size 1` (narrowed to what the self-check covers). kernel_fn's first test is now `(sliding batch-1 decode) or _full_fused_gate(cfg)`, ahead of `_pre_attention_kernel`/FA3. Full batch>1 (splitkv / paired FA3), full verify and all sliding shapes unchanged.
- REQUIRED (1): `_full_fused_self_check()` at import (eager, synthetic data via `_splitkv_synthetic([n])`, before `_build_splitkv`): n in {80000 (kv 1 and 1/4), 5000, 70 (kv 1/4), 1} — covers the scored length (split 63 holds only the new key), empty splits and a single old key. Runs the fused path and master's FA3 path (gate forced off, cloned caches) on the same inputs. Pass requires finite output, rel L2 < 2e-2 vs an FP32 reference with bf16-rounded queries (as the kernel), and new k/v cache rows within 2e-2 of FA3's path. Also prints FA3 vs an FP8-q FP32 reference and fused-vs-FA3 diff (informational: FP8 q vs bf16 q differ by several % on random data, so that diff is not gated). On failure or exception `_FULL_FUSED` stays False and the master FA3 path is kept; it prints `[fullfused] SELF-CHECK FAILED ...`.
- `_splitkv_reference` gained `q_dtype=FP8` (default keeps the existing splitkv self-check bit-for-bit); `q_dtype=torch.bfloat16` rounds after qk_scale_factor.
- Module docstring: one sentence on batch-1 decode.

Uncertainties: (a) 3D `[8, 16, 64]` loads + axis-0 reduction in Triton need layout conversions; compile/regs unverified (no GPU here). (b) The self-check's 80000-length FP32 references allocate ~330 MB transient each and add a few seconds at import. (c) Open questions from the review (pipelining of the dependent page-table load at num_stages=3) untested; record kernel time and 20.5 MB / time bandwidth from the profile.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/full-fused-triton-decode1-gate diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's path; the new gate and launcher branch on `config` only; workspace sizes come from config. The `float(...)` syncs and prints are in the import-time self-check only.
- [x] Nothing derived from `data` is cached across calls: `_FULL_FUSED` is a bool set from synthetic data at import; counters are the existing config-keyed, self-resetting buffer.
- [x] Weights: n/a: this problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (new rows stored E4M3); queries bf16 (finer than E4M3); output bf16.
- [x] Attention sees every key: FULL attends old keys [0, seq-1] (lo = 0, mask tmask only) plus the new key; 64 splits cover the whole range; no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no new imports.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Marker line `# okc: problem=muse_attention_decode` kept; no `triton.autotune`.

Lint:

```
$ python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode
CLEAN: submission.py
```

`python3 -I -m py_compile` OK; `__pycache__` removed.

## Risks / what could fail
medium
Analogue of sliding-fused-triton-decode1-gate (sliding decode_1 16.6 -> 13.1 us) on the full layer, where FA3 at batch 1 spends 30% of the shape in its combine. Risks: no test runs full decode at batch 1, so this path is checked only on the scored call (correctness is checked on every timed call); the implementer should reason through FULL masking against the reference carefully (old keys [0, seq-1] plus the new key, causal trivially satisfied at L=1). The last-ticket CTA reads 64 partials x 4 KB per kv head (256 KB) from L2, ~2-3 us; if that dominates, an ab-tune at S=32 halves it. decode_layer triton-flash-decode-full-paired-kv was flat at decode_1 with a separate combine launch and no fused pre-attention; this removes four of five launches as well as the combine.

## Result (after the run)
- job `af3c6a46badf4e50` (submit), status done, passed True, tests 16/16
- speedup 1.3417x, 95% CI [1.3215, 1.3618], score 2807.1778938172324 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3492076314709118, "ci": [1.328969516998848, 1.3729544014677721], "job": "de486093311d4bb9"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-fused-triton-decode1-gate.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-fused-triton-decode1-gate.patch`; branch `hyp/muse_attention_decode/full-fused-triton-decode1-gate`
