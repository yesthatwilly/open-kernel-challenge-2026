---
problem: muse_decode_layer
slug: full-decode-cuda-splitkv-port
created: 2026-10-24T165154Z
base: master
area: FA3 call
change: >
  Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from
  muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e82154393: full
  decode_16 156.8 -> 123.0 us, decode_32 298.5 -> 257.8 us; attention 1.293x -> 1.347x). Copy
  verbatim from muse_attention_decode/submission.py on master: LOG2E, _SPLITKV_CHUNK,
  _SPLITKV_CPP, _SPLITKV_CUDA (splitkv_attn_kernel = persistent 2xSM grid over 1024-token chunks,
  page indices prefetched to smem, 4-stage cp.async ring of 128-byte K/V rows, fp8->f16 in
  registers, mma.sync m16n8k16, online softmax; splitkv_combine_kernel = LSE merge x v_scale +
  scaleless RMSNorm(64, eps 1e-5) x sigmoid(gate read from qkv at column (Hq+2Hkv)*64 + h*64 with
  qkv.stride(0))), _build_splitkv (load_inline, -O3 -arch=sm_90a, module name
  muse_full_decode_splitkv_v1), _EXT, _SPLITKV_GRID, _SMS/_sm_count, _splitkv_gate and
  _full_decode_splitkv. Compile at import exactly as attention does (guarded by
  torch.cuda.is_available(); on failure _EXT stays None and master path runs everywhere).
  Self-check: attention calls its own kernel_fn on synthetic data; decode cannot (kernel_fn is a
  whole layer), so add _splitkv_self_check() run once in prepare() that builds synthetic inputs
  directly (FP8 q [B,16,64], random FP8 k_cache/v_cache [pages,1,2,64], a shuffled page_table with
  ragged lengths [9000, 1025, 70, 4097, 1023], cache_seqlens int32, k_scale/v_scale 1.0 and 0.25,
  a bf16 qkv [B,2304] supplying the gate) and compares _full_decode_splitkv against an FP32
  reference (gather rows, softmax(q k^T * sm_scale*qk_scale_factor*k_scale), @ v*v_scale, rms-norm
  over 64 eps 1e-5, x sigmoid(gate)) at the device grid and at _SPLITKV_GRID=3; rel L2 < 2e-2 and
  finite or _EXT = None. In decode kernel_fn: compute splitkv = _splitkv_gate(cfg) (config only:
  window 0, q_len 1, 1 < batch <= 64, hq 16, hkv 2, d 64, page_size 1); make paired = window == 0
  and batch > 1 and hkv % 2 == 0 and not splitkv. On the splitkv branch keep master qkv GEMM,
  FusedQKNorm (no RoPE on full layers), the FP8 casts and store_cache byte for byte, take q_fp8 =
  q.view(tokens, hq, d).to(FP8) (contiguous), and replace flash_attn_with_kvcache plus the paired
  pad/unpad copies plus state["o_norm_gate"] with o = _full_decode_splitkv(data, q_fp8,
  qkv).reshape(tokens, hq * d), feeding the unchanged o GEMM. scheduler_metadata is ignored on
  this path. Every other shape (all sliding, full decode_1, every verify block including the
  paired full verify_16/32) keeps master code byte for byte.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "incumbent paired FA3 main 146.8 us + FlashAttnFwdCombine 4.3 + 5 pad/unpad elementwise copies (6.2 + 4.4 us) + FA3 prepare + FusedRMSNormGate, ~165 us of 365 us, floor for the KV gather ~63 us; attention measured kernel A 111.4 + kernel B 7.9 us for the same KV at dist=coherent, so ~40-45 us off: ~320 us, +12-14% on the shape, ~+4% on the decode_16 load"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "incumbent FA3 main 289.0 us + combine + glue ~305 us of 536 us; attention kernel A 245.5 + B 8.6 us: ~45-55 us off, ~485 us, +9-11% on the shape, ~+4% on the decode_32 load"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: batch 1 outside the gate, unpaired FA3 kept
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 outside the gate; paired FA3 kept (stage 2 extends to verify)
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched control
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32
review: proceed
status: done
review_at: 2026-10-24T165606Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T165606Z
base_sha: f3620a153798c4af3feb47a7fc26977635b8a6f3
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/full-decode-cuda-splitkv-port
worktree: .worktrees/muse_decode_layer/full-decode-cuda-splitkv-port
mode: submit
implemented_at: 2026-10-24T165933Z
impl_commit: f6cc1452343c4d93911ad7303c4a5d7878afd2fe
submission_sha256: e75a020601d212ebaa00461ad6dc4445ca25ba188504358caf0c2bdf8c143999
norm_sha256: ddd10ed8feb4284fb885cd0f51c6fa848a93f1de545aeda01bb91b67a228eca3
submission_blob: b83f5ec1976c82d5b45dcd92b6c0818ce04eda9f
lint_warnings: 1
attempt_file: 2026-10-24T181733Z-submit-full-decode-cuda-splitkv-port.md
prepared_at: 2026-10-24T181733Z
job: e789032128df4291
dispatched_at: 2026-10-24T181805Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.186218559298957
speedup_ci: [1.1684252809094726, 1.2040118376884412]
harvested_at: 2026-10-24T184717Z
best_at_harvest: 1.1664628806987323
promotion: null
---

# full-decode-cuda-splitkv-port

## Change

Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e82154393: full decode_16 156.8 -> 123.0 us, decode_32 298.5 -> 257.8 us; attention 1.293x -> 1.347x). Copy verbatim from muse_attention_decode/submission.py on master: LOG2E, _SPLITKV_CHUNK, _SPLITKV_CPP, _SPLITKV_CUDA (splitkv_attn_kernel = persistent 2xSM grid over 1024-token chunks, page indices prefetched to smem, 4-stage cp.async ring of 128-byte K/V rows, fp8->f16 in registers, mma.sync m16n8k16, online softmax; splitkv_combine_kernel = LSE merge x v_scale + scaleless RMSNorm(64, eps 1e-5) x sigmoid(gate read from qkv at column (Hq+2Hkv)*64 + h*64 with qkv.stride(0))), _build_splitkv (load_inline, -O3 -arch=sm_90a, module name muse_full_decode_splitkv_v1), _EXT, _SPLITKV_GRID, _SMS/_sm_count, _splitkv_gate and _full_decode_splitkv. Compile at import exactly as attention does (guarded by torch.cuda.is_available(); on failure _EXT stays None and master path runs everywhere). Self-check: attention calls its own kernel_fn on synthetic data; decode cannot (kernel_fn is a whole layer), so add _splitkv_self_check() run once in prepare() that builds synthetic inputs directly (FP8 q [B,16,64], random FP8 k_cache/v_cache [pages,1,2,64], a shuffled page_table with ragged lengths [9000, 1025, 70, 4097, 1023], cache_seqlens int32, k_scale/v_scale 1.0 and 0.25, a bf16 qkv [B,2304] supplying the gate) and compares _full_decode_splitkv against an FP32 reference (gather rows, softmax(q k^T * sm_scale*qk_scale_factor*k_scale), @ v*v_scale, rms-norm over 64 eps 1e-5, x sigmoid(gate)) at the device grid and at _SPLITKV_GRID=3; rel L2 < 2e-2 and finite or _EXT = None. In decode kernel_fn: compute splitkv = _splitkv_gate(cfg) (config only: window 0, q_len 1, 1 < batch <= 64, hq 16, hkv 2, d 64, page_size 1); make paired = window == 0 and batch > 1 and hkv % 2 == 0 and not splitkv. On the splitkv branch keep master qkv GEMM, FusedQKNorm (no RoPE on full layers), the FP8 casts and store_cache byte for byte, take q_fp8 = q.view(tokens, hq, d).to(FP8) (contiguous), and replace flash_attn_with_kvcache plus the paired pad/unpad copies plus state["o_norm_gate"] with o = _full_decode_splitkv(data, q_fp8, qkv).reshape(tokens, hq * d), feeding the unchanged o GEMM. scheduler_metadata is ignored on this path. Every other shape (all sliding, full decode_1, every verify block including the paired full verify_16/32) keeps master code byte for byte.

## Expected

- `*layer=full load=decode_16 *` faster: incumbent paired FA3 main 146.8 us + FlashAttnFwdCombine 4.3 + 5 pad/unpad elementwise copies (6.2 + 4.4 us) + FA3 prepare + FusedRMSNormGate, ~165 us of 365 us, floor for the KV gather ~63 us; attention measured kernel A 111.4 + kernel B 7.9 us for the same KV at dist=coherent, so ~40-45 us off: ~320 us, +12-14% on the shape, ~+4% on the decode_16 load
- `*layer=full load=decode_32 *` faster: incumbent FA3 main 289.0 us + combine + glue ~305 us of 536 us; attention kernel A 245.5 + B 8.6 us: ~45-55 us off, ~485 us, +9-11% on the shape, ~+4% on the decode_32 load
- `*layer=full load=decode_1 *` flat: batch 1 outside the gate, unpaired FA3 kept
- `*load=verify_*` flat: q_len 8 outside the gate; paired FA3 kept (stage 2 extends to verify)
- `*layer=sliding*` flat: untouched control

## Risk

High: 41 s load_inline compile at import (the judge accepted it in attention), a decode-specific self-check that must be written rather than copied, and decode dist=soft vs attention dist=coherent (the kernel is data-independent, but per-shape lengths differ). Overlaps paired-heads-pad-unpad-triton (running) on full decode_16/32 only: that change still applies to full verify_16/32; at decode_16/32 whichever lands second supersedes the paired glue there. Verify is untouched, so the verify rows are the in-run control. Do not stack on triton-flash-decode-full-* (paused rewrite).

## Milestones

Rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165606Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite. It ports a kernel the judge has already accepted and that passed its self-check (attention 82085a4e82154393: compiled in 41.0 s, rel L2 1.6-1.7e-3), against the same KV geometry (decode_16 full floor 63 us in both problems), and it replaces ~165/~305 us of paired-FA3 work at decode_16/32. Judge it on top_kernels, not the headline: the expected headline move is only ~+1.4%, which is inside run drift.

### Justification
The mechanism is plausible. Full-layer decode at page size 1 is limited by the row gather (LESSONS: 1.3-1.7 TB/s). In muse_attention_decode the hand-written CUDA split-KV kernel beat paired FA3 on this same gather (decode_16 156.8 -> 123.0 us shape, kernel A 111.4 + B 7.9 us). The KV geometry is identical: 16 q heads, 2 KV heads, d 64, page 1, ctx 80k, spread agentic. The attention floor for full decode_16 is 63.2 us, the figure the hypothesis cites. The kernel does not depend on the data, so dist soft vs coherent only changes values, not bytes. In decode_layer's profile, paired FA3 main is 146.8 us (38%) at decode_16 and 289.0 us (52.4%) at decode_32, about what attention's paired FA3 took. That makes the incumbent comparable, and the ~40-55 us saving per full shape is credible. It removes FA3 main, FlashAttnFwdCombine, the paired zero-fill, the pad and unpad copies, the bf16 cast and FusedRMSNormGate, and adds two launches. Load arithmetic from TASK weights: decode_16 load = 46*225.8 + 16*364.9 = 16225 us, and saving ~45 us x16 gives ~4.4%. decode_32 load = 46*251.9 + 16*535.8 = 20160 us, and ~50 us x16 gives ~4.0%. The other four loads are flat, so the geomean is ~+1.4%. The per-shape and per-load expectations are consistent; the headline is below the ~1.5-2.5% drift (LESSONS). Weakest link: it assumes kernel A's attention-measured time carries over unchanged into the layer. In the layer it runs right after the qkv GEMM and FusedQKNorm on a cold L2 (irrelevant at 327 MB of KV). It also runs with 2 CTAs/SM of ~64 KB+ smem, which matches attention's setup. A smaller risk is that the decode-specific self-check is newly written rather than copied: it must build data['config'] with sm_scale, qk_scale_factor, batch, num_q_heads and head_dim for _full_decode_splitkv, and it must reset _SPLITKV_GRID to None in a finally. Run time: the decode run took ~412 s; adding 41 s of compile plus the self-check stays well under the 15-minute limit.

### Confounders on the current kernel
- fa3-hdim128-head-pairing-batched (landed): this is the incumbent being replaced on full decode_16/32. If it were absent, the baseline would be slower unpaired FA3 and the measured gain larger. It is not a confounder for attribution, because the change replaces that path outright on the target shapes, and the paired full verify_16/32 rows keep it as an in-run control.
- post-expert-norm-triton-small-t (landed): it touches T<32 (decode_16), but in the MoE tail after the o GEMM, so the kernels and stream position are disjoint. It does not confound the attention top_kernels rows; it only shifts the decode_16 shape total used to compute the percentage.
- marlin-block-m-cap-16 and sglang-pipeline-parity-router-hilo-m8 (landed): disjoint areas (Marlin block_m does not change at T<=32, router on the side stream after attention). No interaction.
- paired-heads-pad-unpad-triton (in flight, not landed): if it lands first, master's paired glue on full decode_16/32 shrinks by ~10 us. This change's apparent saving against the new master then falls to ~30-45 us. The kernel's own time against FA3 main stays readable either way. A merge must keep `paired = ... and not splitkv`.

### Interference and A/B needs
No companions. This is kind: rewrite, and it is judged on the replaced kernels' own time per shape (splitkv_attn_kernel + splitkv_combine_kernel against FA3 main + combine + pad/unpad + RMSNormGate in top_kernels) against the 63/126 us gather floors. That attribution is direct, and an A/B revert of head pairing would only re-measure a known FA3 variant. The in-run controls are the untouched shapes: every sliding row, full decode_1, and the paired full verify_16/32 rows, which keep the exact incumbent FA3 path. At harvest, check the log tail first: '[splitkv] load_inline failed' or a self-check failure means the run measured the master fallback (DECISIONS 2026-10-24T141733Z).

### Open questions
- Does FusedQKNorm's q come out row-contiguous so that q.view(tokens, hq, d).to(FP8) satisfies the kernel's q.is_contiguous() TORCH_CHECK? (.to() makes a contiguous copy, so this should hold.) Is the k_cache/v_cache from the decode judge contiguous [pages,1,2,64] as the TORCH_CHECK requires?
- The gated tests are 'batch 33, ctx 20000, full, spread mixed, page 1' and 'batch 16, ctx 20000, full, dist soft, page 1'. Both are within 1 < batch <= 64, and they are the correctness gate for this path. Should the self-check also cover batch 33 with a non-power-of-two chunk count?
- If prepare() can be called more than once per process, is the self-check guarded so it runs once, and is _EXT set to None globally on failure before any graph capture?


## Implementation notes

Worktree `.worktrees/muse_decode_layer/full-decode-cuda-splitkv-port`, file `muse_decode_layer/submission.py` only (uncommitted; diff stat `1 file changed, 670 insertions(+), 54 deletions(-)`; with `-w` it is `618 insertions(+), 2 deletions(-)`, i.e. the 52 other "deleted" lines are the master FA3 block re-indented under `else:`, unchanged).

- **Ported verbatim** from `git show HEAD:muse_attention_decode/submission.py` (lines 399-891 there; `diff` of the inserted block against that range is empty): the kernel A/B comment block, `_SPLITKV_CHUNK`, `_SPLITKV_CPP`, `_SPLITKV_CUDA` (splitkv_attn_kernel + splitkv_combine_kernel), `_build_splitkv` (load_inline, `-O3 -arch=sm_90a`, module `muse_full_decode_splitkv_v1`, `[splitkv] load_inline ...` log lines), `_EXT`, `_SPLITKV_GRID`, `_SMS`, `_sm_count`, `_splitkv_gate`, `_full_decode_splitkv`. Plus `LOG2E` and `import time`. Placed after `_post_norm_combine_kernel`, before `prepare`.
- **Import-time compile**: `if torch.cuda.is_available(): _EXT = _build_splitkv()` right after the block, as attention does (on failure `_EXT` stays None and master runs everywhere).
- **Decode-specific self-check (new code)**: `_splitkv_synthetic`, `_splitkv_reference`, `_splitkv_self_check(device)`, `_splitkv_prepare(device)`. Synthetic data only: FP8 q [B,16,64] (RMS-normalized randn then E4M3), randn FP8 k/v caches [pages,1,2,64] divided by kv_scale, shuffled page table with odd stride (max_pages = max(lens)+37), `cache_seqlens` = lens (int32), bf16 qkv [B,2304] supplying the gate, and a `config` dict with every field `_splitkv_gate` and `_full_decode_splitkv` read. Reference is FP32 per the spec (gather rows, softmax(q k^T * sm_scale * qk_scale_factor * k_scale), @ v * v_scale, scaleless RMSNorm over 64 eps 1e-5, x sigmoid(gate)). Cases: lens [9000, 1025, 70, 4097, 1023] at (kv 1.0, device grid), (kv 0.25, grid 3), (kv 1.0, grid 3), as attention did; pass = finite and rel L2 < 2e-2 on every case. `_SPLITKV_GRID` reset to None in `finally`. Each case prints `[splitkv] self-check B=.. kv_scale=.. grid=..: rel L2 .., finite ..`.
- **Run once from prepare()**: `_splitkv_prepare(device)` is called just before prepare's final `torch.cuda.synchronize()`. A module global `_SPLITKV_CHECKED` makes it run once per process (open question 3); on failure it prints `[splitkv] SELF-CHECK FAILED: CUDA split-KV path disabled, master FA3 path kept` and sets the global `_EXT = None` before any graph capture; on success it prints `[splitkv] self-check passed ...`.
- **kernel_fn**: after the unchanged qkv GEMM, FusedQKNorm, FP8 casts and store_cache, `splitkv = _splitkv_gate(cfg)` (config only). On `splitkv`: `q_fp8 = q.view(tokens, hq, d).to(FP8).contiguous()` and `o = _full_decode_splitkv(data, q_fp8, qkv).reshape(tokens, hq * d)`, which feeds the unchanged `_gemm(o, dense["o"])`. Else: the master code, byte for byte apart from indentation, with `paired = window == 0 and batch > 1 and hkv % 2 == 0 and not splitkv` (redundant inside the else, kept as the review asked so a merge with paired-heads-pad-unpad-triton keeps it). scheduler_metadata is not read on the splitkv path.
- Module docstring: one bullet added under "Differs from SGLang".

Deviations / additions, and why:
- `.contiguous()` added after `.to(FP8)`: a no-op when `.to` already made a dense copy, and a guard for the kernel's `q.is_contiguous()` TORCH_CHECK if FusedQKNorm ever returns q as a strided view of qkv (open question 1). Layout facts checked: reference.py builds `k_cache`/`v_cache` with `torch.empty(shape)` (contiguous), and master's own `k_cache.view(-1, hkv * d)` in store_cache already requires that; the gate sits at qkv column (16+4)*64 = 1280 with `qkv.stride(0)` passed explicitly, which matches master's `gate = qkv[:, hq*d + 2*hkv*d:]`; kernel B writes [B, 1, 1024] bf16, and `.reshape(tokens, 1024)` is the same layout as FusedRMSNormGate's output.
- A fourth self-check case was added (open question 2): batch 33, lens `1 + (i*977) % 5000` with lens[5]=1024 and lens[6]=2048 (exact chunk boundaries, a length-1 request), device grid, kv 1.0. This covers the gated test "batch 33, ctx 20000, full" shape class (B > 32, a non-power-of-two chunk total). It checks that inputs give the right output. It does not tune or change anything.
- The synthetic data skips the qkv GEMM, FusedQKNorm and store_cache, because the self-check tests only `_full_decode_splitkv`. Those stages are master code on every path.

Uncertainties: nothing was run here (no GPU). The compile cost (~41 s in attention) comes on top of the decode run (~412 s), still under 15 min. If kernel A hits an illegal address in the self-check, the CUDA context is poisoned and the run fails outright, not falling back. At harvest, read the log tail for `[splitkv]` lines before attributing the timing.

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/full-decode-cuda-splitkv-port diff`:

- [x] CUDA-graph capturable: the splitkv branch in kernel_fn has no `.item()`/`.cpu()`/`synchronize()`/tensor prints; `_full_decode_splitkv` allocates ws_acc/ws_ml/out from shapes (`page_table.shape[1]`, config batch) and the grid from the SM count; the branch is `_splitkv_gate(cfg)` on config plus the module flag `_EXT` fixed in prepare(). The `float(...)`/`bool(...)` host reads are only in `_splitkv_self_check`, called from prepare(), never from kernel_fn.
- [x] Nothing derived from `data` is cached across calls: new module state is `_EXT` (the compiled module), `_SPLITKV_GRID` (None outside the self-check), `_SMS` (device SM count) and `_SPLITKV_CHECKED` (bool). None of them depends on inputs.
- [x] Weights: n/a for this change. prepare()'s weight handling is untouched; the only addition is the synthetic self-check call.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: q goes to E4M3 exactly as master's unpaired FA3 path (`q.view(tokens, hq, d).to(FP8)`), and is widened exactly to f16 in the kernel. K/V stay FP8 in the cache (written by the unchanged store_cache), and are widened in registers. Softmax/accumulators are fp32. P is rounded to f16 for the PV mma (FA3 uses e4m3/bf16 here). The kernel B output is bf16, as FusedRMSNormGate's is.
- [x] Attention sees every key: kernel A walks every 1024-token chunk of `cache_seqlens[b]` for every request. Only positions >= cache_seqlens are masked. There is no top-k, sparse, block-skip or weight-based skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no new imports beyond `time` and `torch.utils.cpp_extension.load_inline` (the same build the judge accepted in muse_attention_decode job 82085a4e82154393; nvcc/ninja run inside torch's extension builder at import). The lint process/network/harness rules are clean.
- [x] All work happens inside the timed call; no output caching or replay: the whole attention for the step runs in kernel_fn (kernels A and B) on the step's own q, cache and page table. The import-time compile and the prepare()-time self-check produce no output that is used later.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept.
- [x] No `triton.autotune`. No new Triton kernels at all.
- [x] Every non-gated shape (all sliding, full decode_1, every verify incl. paired full verify_16/32) runs the master code unchanged: `git diff -w` shows only two removed lines, the docstring's closing line (extended) and the `paired =` line (+ `and not splitkv`).

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L884: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

(The first lint pass flagged `error network` on the word "requests" in the self-check docstring; reworded to "sequences".) `python3 -I -m py_compile` OK; __pycache__ removed.

## Result

- job `e789032128df4291` (submit), status done, passed True, tests 13/13
- speedup 1.1862x, 95% CI [1.1684, 1.2040], score 19069.251465234505 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-cuda-splitkv-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-cuda-splitkv-port.patch`; branch `hyp/muse_decode_layer/full-decode-cuda-splitkv-port`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 145.83 | 144.97 | 1.0089 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 156.95 | 156.49 | 1.0095 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 225.79 | 230.07 | 0.9939 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 364.88 | 329.13 | 1.1276 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 251.94 | 260.43 | 0.9966 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 535.78 | 490.19 | 1.1026 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 200.79 | 202.99 | 1.0049 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 218.06 | 220.46 | 0.9926 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 462.37 | 435.66 | 1.0271 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 603.87 | 590.55 | 1.0285 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 639.36 | 643.87 | 1.0085 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 932.92 | 909.53 | 0.9948 | flat |

gained on full decode_16, decode_32. Expected hit: 5/5. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_decode_layer "<what to fix or tune>" --kind rewrite --depends-on full-decode-cuda-splitkv-port --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-cuda-splitkv-port`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

