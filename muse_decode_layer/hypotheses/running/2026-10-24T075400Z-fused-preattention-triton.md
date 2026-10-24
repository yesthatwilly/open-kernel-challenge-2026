---
problem: muse_decode_layer
slug: fused-preattention-triton
created: 2026-10-24T075400Z
base: master
area: pre-attention norm+rope
change: >
  Port muse_attention_decode's fused pre-attention Triton kernel (main tree
  muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x there) into
  kernel_fn, replacing on every shape the five launches between the qkv GEMM and
  flash_attn_with_kvcache: state['qk_norm'](...) (FusedQKNorm, with RoPE on sliding layers), the
  optional k/v divide by the cache scales, k.to(FP8), v.to(FP8), store_cache(...) and
  q.view(T,16,64).to(FP8). One launch, grid (T, 16+2), num_warps=1, one program per (token, head):
  heads 0-15 RMS-normalize (scaleless, eps 1e-5) the q head read from the qkv row (stride
  qkv.stride(0)), apply interleaved RoPE when cfg['rope_theta'] is set and store FP8 into a fresh
  q buffer [T, 16, 64] float8_e4m3fn; heads 16-17 normalize (+RoPE) the k head, store (k /
  k_scale) as FP8 into k_cache.view(-1, 128) row cache_loc[t] at column (h-16)*64, and store the
  matching v head (v / v_scale, no norm or RoPE) into v_cache at the same row. One deliberate
  difference from the sibling: RoPE cos/sin are read from master's state['cos_sin'] table (row
  positions[t], cos in columns 0-31, sin in 32-63) instead of recomputed in-kernel, so the
  rotation is the table SGLang's FusedQKNorm uses. The FP8 q goes to flash_attn_with_kvcache
  unchanged otherwise; gate and v are still slices of qkv. FA3, the output norm-gate and
  everything downstream stay as on master.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: 5 serialized launches -> 1 on a latency-bound 169/181 us shape; the sibling saved 3.5-6.4 us for this exact 5->1 fusion (~2-3.5%)
  - shapes: "*load=verify_1*"
    direction: faster
    why: T=8, same 5->1 fusion on 229-251 us shapes (~1.5-2.5%)
  - shapes: "*load=decode_16*"
    direction: flat
    why: a few us on 265-440 us shapes, under the drift
  - shapes: "*load=decode_32*"
    direction: flat
    why: as decode_16
  - shapes: "*load=verify_16*"
    direction: flat
    why: 520-714 us shapes
  - shapes: "*load=verify_32*"
    direction: flat
    why: 713-1050 us shapes
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T124334Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T124334Z
base_sha: 2d766b8ec8494820bc79165703df0b509a3ef4b5
base_blob: 3a358375a50dcc6e19437992f7c610431582ea72
branch: hyp/muse_decode_layer/fused-preattention-triton
worktree: .worktrees/muse_decode_layer/fused-preattention-triton
mode: check
implemented_at: 2026-10-24T124616Z
impl_commit: fc6abaceb9f672a692fdf540a79cbe6ba771e03d
submission_sha256: f4a968ff49825f1c5e0bc8f368424645afa047f708b1ab632a9670436cdfd22c
norm_sha256: 2d9ee35b6e88b1034ea306408f6dd2b70b44f0e4012a35935ed40034f2bc9ff7
submission_blob: 81fe2e3fbef85c828ca0b27215cd93988531b7e6
lint_warnings: 1
attempt_file: 2026-10-24T124616Z-check-fused-preattention-triton.md
prepared_at: 2026-10-24T124616Z
job: c7c26e2fd8b142a7
dispatched_at: 2026-10-24T124625Z
---

# fused-preattention-triton

## Change

Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x there) into kernel_fn, replacing on every shape the five launches between the qkv GEMM and flash_attn_with_kvcache: state['qk_norm'](...) (FusedQKNorm, with RoPE on sliding layers), the optional k/v divide by the cache scales, k.to(FP8), v.to(FP8), store_cache(...) and q.view(T,16,64).to(FP8). One launch, grid (T, 16+2), num_warps=1, one program per (token, head): heads 0-15 RMS-normalize (scaleless, eps 1e-5) the q head read from the qkv row (stride qkv.stride(0)), apply interleaved RoPE when cfg['rope_theta'] is set and store FP8 into a fresh q buffer [T, 16, 64] float8_e4m3fn; heads 16-17 normalize (+RoPE) the k head, store (k / k_scale) as FP8 into k_cache.view(-1, 128) row cache_loc[t] at column (h-16)*64, and store the matching v head (v / v_scale, no norm or RoPE) into v_cache at the same row. One deliberate difference from the sibling: RoPE cos/sin are read from master's state['cos_sin'] table (row positions[t], cos in columns 0-31, sin in 32-63) instead of recomputed in-kernel, so the rotation is the table SGLang's FusedQKNorm uses. The FP8 q goes to flash_attn_with_kvcache unchanged otherwise; gate and v are still slices of qkv. FA3, the output norm-gate and everything downstream stay as on master.

## Expected

- `*load=decode_1*` faster: 5 serialized launches -> 1 on a latency-bound 169/181 us shape; the sibling saved 3.5-6.4 us for this exact 5->1 fusion (~2-3.5%)
- `*load=verify_1*` faster: T=8, same 5->1 fusion on 229-251 us shapes (~1.5-2.5%)
- `*load=decode_16*` flat: a few us on 265-440 us shapes, under the drift
- `*load=decode_32*` flat: as decode_16
- `*load=verify_16*` flat: 520-714 us shapes
- `*load=verify_32*` flat: 713-1050 us shapes

## Risk

This is the superset of cache-store-fused-fp8-cast (3 of 4 cast/store launches removed, came back flat at 0.996/1.000 on decode_1): it additionally folds FusedQKNorm, the launch the earlier run left in, which is what the muse_attention_decode sibling fused to get its 3.5-6.4 us. Built on master, not on that neutral branch (the fused kernel replaces its code entirely). Expected per-shape gains (~2-3%) sit near the 1.5% drift, so read the T=1 and T=8 rows individually. Correctness: q is rounded fp32 -> FP8 directly instead of fp32 -> bf16 -> FP8 (allowed: queries may be rounded to FP8); k likewise; the kv_scale 0.015625 test and page-64 tests exercise the scale divide and the flat-row addressing (loc * 2 + kh) * 64. If triton-flash-decode-full-paired-kv lands first, its q input is this kernel's FP8 q buffer.

## Adversarial review

Reviewed 2026-10-24T124334Z by an adversarial subagent (model opus).

**Verdict: proceed.** Master has not made this moot: kernel_fn still runs FusedQKNorm, the k/v FP8 casts, store_cache and the q FP8 cast as separate launches, and the change is a strict superset of the neutral cache-store-fused-fp8-cast because it also folds FusedQKNorm. The gain it can show is small, about 2-3% on the batch-1 shapes and +0.5-0.8% on the headline, inside the CI, so read it on the decode_1 and verify_1 rows. The spec predates head pairing, so the implementer has to say what the paired branch does with the new FP8 q.

### Justification
The mechanism is plausible. At T=1 and T=8 the layer is a serial chain of latency-bound launches inside a CUDA graph. The 172.8/182.1 us decode_1 and 238/248 us verify_1 shapes sit 4x above their floors (40-66 us), and launch overhead is a large part of that gap. The sibling measured 3.5-6.4 us for this same 5->1 fusion. If those microseconds carry over unchanged to the serialized layer graph (no overlap hides them on the main stream before FA3), the claimed ~2-3.5% on decode_1 and ~1.5-2.5% on verify_1 follows. The expected directions match TASK.md. decode_1 and verify_1 are where a few us matter, and every batch-16/32 shape is 260-1070 us and Marlin/FA3 dominated, so flat is the right call there. Geomean arithmetic: two of six loads gaining ~2-3% gives about +0.5-0.8% on the headline, which is inside the ±1.5% CI. The run is therefore only readable per shape, or from the profile: the replaced kernels' rows (FusedQKNorm Triton, the unrolled_elementwise casts, store_cache, the q cast) should disappear and one ~2-3 us Triton row should appear. Weakest link: LESSONS records that tiny-kernel fusions were flat on decode_layer shapes. In particular cache-store-fused-fp8-cast, which removed 3 of these 4 cast/store launches, came back within noise (1.011/1.009 on decode_1). So the case rests almost entirely on FusedQKNorm (plus the last q cast) being worth ~2-4 us on its own in this layer. The sibling's savings were measured on 17-36 us attention-only shapes, where L2 state and PDL overlap differ. In the full layer the attention input comes from the qkv GEMM, and only qkv is warm, not the whole working set. Expect the low end of 3.5-6.4 us, and a decode_1 result anywhere from flat to +3%.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched: on full layers with batch>1 (full decode_16/32, verify_16/32), master builds q_attn with torch.zeros plus two strided copies from the bf16 q, and that copy also does the BF16->E4M3 cast. With the spec as written, the fused kernel emits an unpadded FP8 q [T,16,64], and the pad path stays as zeros plus two copies (now FP8->FP8). On those shapes the saving is therefore only FusedQKNorm, the k/v casts and the store. The pad launches stay, so they read flat, which agrees with the spec's 'flat'. Batch-1 shapes are never paired, so the decode_1 and verify_1 rows that carry the hypothesis are not confounded. If the implementer instead ports the sibling's current PAIRED branch (padded q written straight from the kernel), the paired shapes would also lose the zeros and copy launches. That is a separate sub-change and should not be folded in silently, because it would mix two effects on the full batch>1 rows.
- sglang-pipeline-parity-router-hilo-m8: it touches only the router and the boundary kernel's router_dtype at T=8, downstream of attention. No interaction with the pre-attention chain. Its FusedQKNorm/cos_sin table is the very thing being replaced, and the spec reads the same table, so RoPE numerics stay aligned with it.
- marlin-block-m-cap-16: it affects only the MoE GEMMs at T>=32. No interaction.

### Interference and A/B needs
No companion is warranted. The predicted gain sits on the batch-1 shapes, which no landed hypothesis touches in the pre-attention path. Pairing applies only at batch>1 and the router change is downstream. A gain or a loss on decode_1/verify_1 can be attributed to the fusion directly, and the profile's replaced-kernel rows confirm it independently of drift. An ab-revert of head pairing would only change the full batch>1 shapes, which are predicted flat, so it cannot disambiguate anything. A control splitting FusedQKNorm from the cast/store fusion is already effectively on file: cache-store-fused-fp8-cast measured the cast/store part (flat), so this run's delta over it is the FusedQKNorm fold. The real ambiguity is drift: ~2% effects against 1.5-2.5% per-shape drift. Settle that with the profile's kernel times rather than a second run.

### Open questions
- Paired branch: the spec says the FP8 q goes to FA3 'unchanged otherwise', but it was written before head pairing landed. Confirm that the implementation feeds the kernel's FP8 q [T,16,64] into master's existing zeros-plus-two-copies pad on the paired path, and does not adopt the sibling's PAIRED in-kernel write. The PAIRED write is the obvious follow-up hypothesis (depends_on this one) for full batch>1 shapes, where it removes two or three more launches; attention_decode got 1.244x from it.
- Exactly which launches disappear at T=1 depends on whether k/v are divided (only when kv_scale != 1) and on whether q.view().to(FP8) is still a separate launch on the unpaired path. Count the removed kernel rows in the decode_1 profile so the measured us per removed launch can go into LESSONS whatever the outcome.
- Does FusedQKNorm's RoPE use the same interleaved pairing (2i, 2i+1) and the table layout cos[0:32] | sin[32:64] indexed by pair i? The spec assumes so from prepare(). A mismatch would show up only on sliding layers (rope_theta set) and should fail the sliding tests loudly rather than subtly. Verify it against sglang's fused_qk_norm source before submitting.
- num_warps=1 with grid (T, 18) is fine at T<=8. At T=256 (verify_32) it is 4,608 tiny programs. Still cheap, but check that the verify_32 row does not regress against the separate vectorized casts.


## Implementation notes

Worktree `.worktrees/muse_decode_layer/fused-preattention-triton`, file `muse_decode_layer/submission.py` only (uncommitted; diff stat `1 file changed, 83 insertions(+), 21 deletions(-)`).

- New `@triton.jit _pre_attention_kernel` (placed after `prepare()`), ported from `muse_attention_decode/hypotheses/done/gain/fused-preattention-triton.patch`. Grid `(T, hq + hkv)` = `(T, 18)`, `num_warps=1`, one program per (token, head). Heads 0-15: scaleless RMSNorm (eps `RMS_EPS` = 1e-5) of the q head read from the qkv row at `t * qkv.stride(0) + h*64`, optional interleaved RoPE, stored as `tl.float8e4nv` into a fresh `q = torch.empty((T, 16, 64), FP8)`. Heads 16-17: same norm (+RoPE) on the k head, stored as `(k / k_scale)` FP8 into the cache at flat offset `(cache_loc[t] * 2 + kh) * 64`, plus the matching v head (no norm/RoPE) as `(v / v_scale)` FP8 into `v_cache` at the same offset.
- Spec deviation from the sibling (as the spec requires): RoPE reads `state["cos_sin"]` (row `positions[t]`, cos columns 0-31, sin 32-63, row stride D=64) instead of computing `inv_freq`/`tl.cos`/`tl.sin` in-kernel. `ROPE` is a constexpr from `bool(cfg["rope_theta"])`, so the table pointer is passed but unread on full layers.
- kernel_fn: removed `state["qk_norm"](...)`, the `v` slice, the `k.reshape`, the `if scaled:` k/v divides, the k/v `.to(FP8)` casts and `store_cache(...)`; the unpaired branch now passes the kernel's FP8 `q` straight to FA3 (the `q.view(...).to(FP8)` cast is gone). The k/v divide now always runs in-kernel (dividing by 1.0 on scored shapes, exact). `scaled` is kept: it still drives FA3's `k_descale`/`v_descale`.
- Paired branch (full layers, batch>1), per the brief and the review's open question: master's `torch.zeros` + two strided pad copies are kept exactly as they were; they now read the kernel's FP8 `q` [T,16,64] (FP8 -> FP8 copies instead of BF16 -> FP8). No in-kernel paired write. Comment updated to say so.
- prepare(): removed the `"qk_norm": FusedQKNorm(...)` entry and the `FusedQKNorm` / `store_cache` imports (now unused); added `import triton`, `import triton.language as tl`. `state["cos_sin"]` is unchanged. The module docstring's pipeline table is updated for the new launch.
- Open question "same RoPE convention as FusedQKNorm": sglang's `fused_qk_norm` source is not on this node, so it could not be read directly. Indirect evidence: the reference `_rope` is interleaved (pairs 2i, 2i+1) with `inv_freq = theta^(-2i/64)` and angle `pos * inv_freq`; `prepare()` builds `cos_sin` as `cat(cos(pos*inv_freq), sin(pos*inv_freq))` with the same inv_freq; master (FusedQKNorm + this table) passes the sliding tests; baseline.py documents it as "interleaved RoPE". So the kernel's pairing `(2i, 2i+1)` with `cos[i]`, `sin[i]` matches the reference whatever FusedQKNorm does internally. A mismatch would fail the sliding tests loudly.
- Numerics: q and k go fp32 -> FP8 directly (master: fp32 -> bf16 -> FP8), allowed ("queries may be rounded to FP8"); v is bf16 -> fp32 / v_scale -> FP8.
- Not verifiable here (no GPU/Triton): compile, correctness, and the verify_32 (T=256, 4,608 programs) regression check the review asks for; read those from the run and its profile. Count the removed kernel rows in the decode_1 profile (FusedQKNorm, casts, store_cache, q cast) for LESSONS.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/fused-preattention-triton diff`:

- [x] CUDA-graph capturable: the new launch has no `.item()`/`.cpu()`/`synchronize()`/tensor prints; scales, positions and cache_loc are read in-kernel as device pointers; the grid and constexprs come from shapes and `config` only (`tokens`, `hq`, `hkv`, `d`, `cfg["rope_theta"]`). The one `torch.empty` for q has a shape-only size, as master's allocations do. No `triton.autotune`.
- [x] Nothing derived from `data` is cached across calls: q is a fresh per-call buffer; `state` gains nothing (it loses `qk_norm`); `cos_sin` is built in `prepare()` from constants only.
- [x] Weights: n/a: untouched; no new weight repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (same rows/offsets as store_cache wrote); queries FP8 E4M3 (fp32 -> E4M3 directly), not coarser. Gate, attention output and everything downstream unchanged.
- [x] Attention sees every key: FA3 call, window and page table unchanged; nothing skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: only `triton`/`triton.language` imports added; two sglang imports removed.
- [x] All work happens inside the timed call: the norm, RoPE, casts and cache writes all run in kernel_fn each call; no output caching or replay.
- [x] First line still `# okc: problem=muse_decode_layer`.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L249: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/fused-preattention-triton/muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed.
