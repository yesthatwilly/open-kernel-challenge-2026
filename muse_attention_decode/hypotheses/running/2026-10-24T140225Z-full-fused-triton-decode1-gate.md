---
problem: muse_attention_decode
slug: full-fused-triton-decode1-gate
created: 2026-10-24T140225Z
base: master
area: full-layer attention (batch-1 decode gate)
change: >
  Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and
  q_len == 1) through master _sliding_fused_kernel via _sliding_forward, replacing the unpaired
  path there (_pre_attention_kernel + prepare_varlen_num_blocks + FA3 main 14.8 us +
  FlashAttnFwdCombine 7.2 us + rms_norm_gate). Kernel edits, all behind new constexprs so the
  sliding instantiation compiles to the same code: (a) FULL: tl.constexpr (= window == 0): lo = 0
  instead of max(seq - (WINDOW - 1), 0), and the old-key validity is tmask only (every old key t <
  seq <= p_row is attended; no window lower edge); ROPE is already off for full layers (rope_theta
  0) and the new key/value is built, stored at cache_loc and attended from registers exactly as on
  sliding; (b) the split combine loops: keep tl.static_range(S) when S <= 16 (sliding, unchanged)
  and use a runtime range(S) loop when S > 16 (a constexpr branch), so a deep split does not
  unroll 64 copies. _sliding_forward chooses, for FULL only, S = 64 and BLOCK_N = 128 (80000 old
  keys -> 1250 per split, ~10 tiles of 128), grid (1, 2, 64) = 128 CTAs (about one per SM),
  num_warps=4, num_stages=3; workspace ws_acc [128, 16, 64] fp32 (512 KB) and ws_ml from config;
  the counters come from _counters(1, 2) as today. Sliding keeps S from _num_splits, BLOCK_N=64,
  num_stages=2. scheduler_metadata is not read on this path. Full batch>1 (paired FA3), full
  verify_1 and every sliding shape are unchanged.
expected:
  - shapes: "*layer=full load=decode_1*"
    direction: faster
    why: incumbent 26.99 us shape / 24.4 us kernels against a 4.3 us floor (20 MB of KV); FA3 combine alone is 7.2 us and prepare 2.8 us; one launch with 128 CTAs streaming ~160 KB each with 3-stage pipelining plus an in-kernel 64-way combine should be ~12-16 us; full is 42% of the decode_1 load, ~1.1-1.15x on it
  - shapes: "*layer=sliding*"
    direction: flat
    why: same constexprs and code for sliding instantiations
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch>1 keeps the paired FA3 path
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 untouched
risk: medium
depends_on: sliding-fused-triton-decode1-gate
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164553Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T164553Z
base_sha: bcc7ea4a50f9cbe47528f96e0ab15b661666ecdf
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/full-fused-triton-decode1-gate
worktree: .worktrees/muse_attention_decode/full-fused-triton-decode1-gate
mode: null
implemented_at: 2026-10-24T164956Z
impl_commit: 96908abef931000bcc9d79b0d10a47c895f466f4
submission_sha256: dd8d09f5010575141d2e2b4a424688fd1bbd4c244f9c6e9cd2b79a67005ee9af
norm_sha256: dc3f357b4467875a82260c8b98e78b3c66b06d2f13f8b70a001c39505c25f181
submission_blob: dbe93e3308cb9143edde3f8784fbffce594bf70a
lint_warnings: 0
---

# full-fused-triton-decode1-gate

## Change

Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forward, replacing the unpaired path there (_pre_attention_kernel + prepare_varlen_num_blocks + FA3 main 14.8 us + FlashAttnFwdCombine 7.2 us + rms_norm_gate). Kernel edits, all behind new constexprs so the sliding instantiation compiles to the same code: (a) FULL: tl.constexpr (= window == 0): lo = 0 instead of max(seq - (WINDOW - 1), 0), and the old-key validity is tmask only (every old key t < seq <= p_row is attended; no window lower edge); ROPE is already off for full layers (rope_theta 0) and the new key/value is built, stored at cache_loc and attended from registers exactly as on sliding; (b) the split combine loops: keep tl.static_range(S) when S <= 16 (sliding, unchanged) and use a runtime range(S) loop when S > 16 (a constexpr branch), so a deep split does not unroll 64 copies. _sliding_forward chooses, for FULL only, S = 64 and BLOCK_N = 128 (80000 old keys -> 1250 per split, ~10 tiles of 128), grid (1, 2, 64) = 128 CTAs (about one per SM), num_warps=4, num_stages=3; workspace ws_acc [128, 16, 64] fp32 (512 KB) and ws_ml from config; the counters come from _counters(1, 2) as today. Sliding keeps S from _num_splits, BLOCK_N=64, num_stages=2. scheduler_metadata is not read on this path. Full batch>1 (paired FA3), full verify_1 and every sliding shape are unchanged.

## Expected

- `*layer=full load=decode_1*` faster: incumbent 26.99 us shape / 24.4 us kernels against a 4.3 us floor (20 MB of KV); FA3 combine alone is 7.2 us and prepare 2.8 us; one launch with 128 CTAs streaming ~160 KB each with 3-stage pipelining plus an in-kernel 64-way combine should be ~12-16 us; full is 42% of the decode_1 load, ~1.1-1.15x on it
- `*layer=sliding*` flat: same constexprs and code for sliding instantiations
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch>1 keeps the paired FA3 path
- `*load=verify_*` flat: q_len 8 untouched

## Risk

Analogue of sliding-fused-triton-decode1-gate (sliding decode_1 16.6 -> 13.1 us) on the full layer, where FA3 at batch 1 spends 30% of the shape in its combine. Risks: no test runs full decode at batch 1, so this path is checked only on the scored call (correctness is checked on every timed call); the implementer should reason through FULL masking against the reference carefully (old keys [0, seq-1] plus the new key, causal trivially satisfied at L=1). The last-ticket CTA reads 64 partials x 4 KB per kv head (256 KB) from L2, ~2-3 us; if that dominates, an ab-tune at S=32 halves it. decode_layer triton-flash-decode-full-paired-kv was flat at decode_1 with a separate combine launch and no fused pre-attention; this removes four of five launches as well as the combine.

## Adversarial review

Reviewed 2026-10-24T164553Z by an adversarial subagent (model opus).

**Verdict: proceed.** Master has not made this moot: _splitkv_gate needs 1 < batch, so full decode_1 still runs unpaired FA3 at 27.9 us (main 14.5, combine 7.0, pre-attn 1.3, plus prepare and rms_norm_gate). Reusing the proven fused batch-1 kernel there is a sound idea and the result reads directly from one shape. Because this is a scored submit and no test reaches the path, the implementer should add an import-time self-check with a fallback, and should not run the 64-way combine as a serial runtime loop.

### Justification
The mechanism is plausible. At batch 1 the full layer costs 27.9 us against a 4.3 us floor. About 10 us of that is FA3 combine plus prepare plus separate launches, and the fused kernel removes it, as it did on sliding decode_1 (16.6 -> 13.1 us). The masking change is correct against reference.py. seq_lens is the count of old tokens, positions = seq_lens + j, and the reference attends keys_past at 0..seq-1 plus the new entries. So FULL means lo = 0 and valid = tmask, and the new key and value are attended from registers as on sliding. Causality is trivial at L=1. The chunk arithmetic checks out at seq = 80000: chunk = 1280, split 63 has no old keys but holds the new key, so m_max stays finite. At short seq the empty partials get weight 0. ROPE is off on full layers. kv_scale goes through device pointers as on sliding, which a test already covers. Bf16 q is finer than E4M3, so it is allowed. The 512 KB workspace and the shared counters follow config only. The direction claims match TASK.md: only full decode_1 enters the gate, and sliding instantiations compile the same if FULL=False and S<=16 keep static_range, BLOCK_N=64 and num_stages=2. The expected 12-16 us is optimistic, though. LESSONS records that Triton never got past ~2.1 TB/s on this page-size-1 gather (1.3-1.7 TB/s typical), so streaming 20 MB alone takes ~10-13 us. On top of that, the sliding fused kernel already spends 10.3 us on only ~0.5 MB of KV, which means ~7-9 us of fixed prologue, fence, ticket, combine and epilogue cost. A realistic landing is 17-22 us. That is still a gain: at 20 us the decode_1 load goes 1085.8 -> ~960 (1.13x), about +2% on the 6-load geomean. At 14 us it would be ~1.26x on the load. WEAKEST LINK: the proposed runtime range(S) combine for S=64. One CTA would make 2 x 64 serial, dependent L2 round trips of [16,64] fp32 partials. At ~0.2-0.4 us each that is 25-50 us on the critical path and would erase the gain. Keep it unrolled: tl.static_range(64) only costs compile time. Better still, load ws_ml as one [S, M] tile (4 KB) and compute m_max, the weights and l_sum vectorized, then stream acc with independent, unrolled loads. Or use S=32 as the hypothesis itself suggests. Secondary: one program per kv head fetches 64-byte halves of each 128-byte row, so there are twice as many gather requests as in the CUDA split-KV design. That is acceptable at 20 MB.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate: this change edits the same kernel and the same _sliding_forward launcher, and the two shapes share the _counters(1, 2) buffer. Any sliding decode_1 movement means the edit leaked into the sliding instantiation (FULL=False and S<=16 must keep today's code, BLOCK_N=64 and num_stages=2). That shows up as a per-shape check, not as ambiguity. The shared counter is safe because every launch resets it to 0 and the shapes replay sequentially.
- full-decode-cuda-gather-splitkv: disjoint gate (batch > 1). The batch==1 test must come before its gate so that full decode_1 never reaches _pre_attention_kernel or FA3. It does not confound the measurement, and cold L2 on each replay removes cache interaction.
- paired-heads-launch-cleanup-ab-tune-batch-gate / fa3-hdim128-head-pairing / paired-heads-launch-cleanup: batch 1 already runs unpaired, so the incumbent being replaced is plain FA3. No interaction.
- fused-preattention-triton: on this shape it is replaced by _norm_rope plus the in-kernel cache store, the same code sliding decode_1 already validates. Not a confounder.

### Interference and A/B needs
No companion is needed. The change touches one scored shape (full decode_1) through a config-only gate, and every other shape is byte-identical code. Its effect reads unambiguously from that shape's row and from _sliding_fused_kernel's own time in the profile against the 24.4 us of FA3 main, combine, prepare, pre-attn and norm-gate it replaces. A gain or a loss there can only come from this kernel. An S=32 ab-tune would only be useful as a follow-up if the profile shows the combine dominating, and a single scored submit cannot carry it anyway. Widening _splitkv_gate to batch >= 1 is a competing alternative (79 chunk items over 264 CTAs, 3 launches), not a disambiguating control, so it belongs in the queue as its own hypothesis.

### Open questions
- Does Triton's pipeliner actually overlap the dependent page-table load with K/V in the runtime range(start, end, 128) loop at num_stages=3 (BLOCK_N=128: 3 stages x 16 KB = 48 KB smem)? If not, ~10 dependent round trips per CTA will set the floor near 10 us on their own.
- Would one program per (b, split) covering both kv heads, reading each 128-byte row once, beat two programs per row-half here? The CUDA split-KV design reached ~2.95 TB/s that way.
- Is widening _splitkv_gate to batch >= 1 (a one-line change to a kernel that is already self-checked) a cheaper route to the same ~10 us? It would cost 3 launches, against the fused kernel's 1.
- Expected 12-16 us versus a realistic 17-22 us: record the replaced kernel's own time and the achieved bandwidth (20.5 MB / kernel time) so a modest gain is not read as a failure.


## Implementation notes

Worktree `.worktrees/muse_attention_decode/full-fused-triton-decode1-gate`, file `muse_attention_decode/submission.py` only (uncommitted; +130/-33).

- `_sliding_fused_kernel`: new constexpr `FULL`. When FULL, `lo = 0` (static branch) and the old-key validity is `tmask[None, :]` only; the sliding branch keeps `max(seq - (WINDOW - 1), 0)` and the per-row lower edge verbatim. New key/value build, cache store and in-register attend are untouched (ROPE is off because rope_theta is 0).
- Combine, deviation from the md per the review's REQUIRED (2): no runtime `range(S)` loop. A static `if S > 16` branch loads all S (m, l) partials as one `[S, M]` tile (S=64, M=16 -> 4 KB each), computes m_max, the weights and l_sum vectorized, then streams acc as 8 unrolled, independent `[CS=8, M, D]` 3D loads (32 KB each) reduced over the split axis with `tl.sum(axis=0)`. `S <= 16` keeps the original `tl.static_range(S)` code (re-indented only), so sliding instantiations (FULL=False, S<=16, BLOCK_N=64, num_stages=2) compile the same. `CS: tl.constexpr = 8` declared at the kernel top (compile-time only).
- `_sliding_forward`: `full = window == 0` -> S = `_FULL_SPLITS` (64), BLOCK_N=128, num_stages=3; else S = `_num_splits`, BLOCK_N=64, num_stages=2. num_warps=4 both. Workspace ws_acc [128, 16, 64] fp32 / ws_ml [128, 2, 16] and `_counters(1, 2)` as for sliding. scheduler_metadata not read.
- New `_full_fused_gate(cfg)`: `_FULL_FUSED and window == 0 and batch == 1 and q_len == 1 and tp8 (16/2) and head_dim 64 and page_size 1` (narrowed to what the self-check covers). kernel_fn's first test is now `(sliding batch-1 decode) or _full_fused_gate(cfg)`, ahead of `_pre_attention_kernel`/FA3. Full batch>1 (splitkv / paired FA3), full verify and all sliding shapes unchanged.
- REQUIRED (1): `_full_fused_self_check()` at import (eager, synthetic data via `_splitkv_synthetic([n])`, before `_build_splitkv`): n in {80000 (kv 1 and 1/4), 5000, 70 (kv 1/4), 1} — covers the scored length (split 63 holds only the new key), empty splits and a single old key. Runs the fused path and master's FA3 path (gate forced off, cloned caches) on the same inputs. Pass requires finite output, rel L2 < 2e-2 vs an FP32 reference with bf16-rounded queries (as the kernel), and new k/v cache rows within 2e-2 of FA3's path. Also prints FA3 vs an FP8-q FP32 reference and fused-vs-FA3 diff (informational: FP8 q vs bf16 q differ by several % on random data, so that diff is not gated). On failure or exception `_FULL_FUSED` stays False and the master FA3 path is kept; it prints `[fullfused] SELF-CHECK FAILED ...`.
- `_splitkv_reference` gained `q_dtype=FP8` (default keeps the existing splitkv self-check bit-for-bit); `q_dtype=torch.bfloat16` rounds after qk_scale_factor.
- Module docstring: one sentence on batch-1 decode.

Uncertainties: (a) 3D `[8, 16, 64]` loads + axis-0 reduction in Triton need layout conversions; compile/regs unverified (no GPU here). (b) The self-check's 80000-length FP32 references allocate ~330 MB transient each and add a few seconds at import. (c) Open questions from the review (pipelining of the dependent page-table load at num_stages=3) untested; record kernel time and 20.5 MB / time bandwidth from the profile.

## Rules check

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
