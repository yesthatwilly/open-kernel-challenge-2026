---
problem: muse_attention_decode
slug: full-decode1-cuda-splitkv-chunk640
created: 2026-10-24T183847Z
base: master
area: full-layer attention (batch-1 decode gate)
change: >
  Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer
  decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's CUDA split-KV path
  instead of FA3. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel
  (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e <
  CHUNK; e += THREADS)`); kernel bodies otherwise byte for byte. (2) splitkv_combine_kernel gets
  template <int CHUNK, int NW> (red arrays [NW], loop strides NW * 32 and NW, __launch_bounds__(NW
  * 32)). (3) Host functions take an int64 chunk argument (update _SPLITKV_CPP): 1024 -> <1024>
  attn and <1024, 4> combine (today's code); 640 -> <640> attn and <640, 8> combine; anything else
  TORCH_CHECK fails. Module renamed muse_full_decode_splitkv_v2. (4) Python: _splitkv_gate allows
  1 <= batch; _full_decode_splitkv uses chunk = 640 if batch == 1 else 1024, max_chunks =
  ceil(page_table.shape[1] / chunk), grid unchanged (2 x SM count; at 80K keys only 125 items
  exist, so 125 CTAs get one 10-tile item each and the rest exit at the `bx >= total` check). (5)
  _splitkv_self_check adds batch-1 cases lens [80000] and [5000], kv_scale 1 and 0.25, grid device
  and 3, rel L2 < 2e-2. Print '[splitkv] batch-1 chunk 640 path ON/OFF' once at import.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "incumbent 27.3 us (FA3 14.6 + combine 7.1 + pre 1.3 + norm_gate 1.1). 125 CTAs x 160 KB each with 3 tiles in flight per CTA; ~8-10 us for kernel A and only 125 partials for an 8-warp combine (~1.5-2.5 us): ~14-17 us shape"
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside the gate
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: batch > 1 keeps chunk 1024
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184117Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T184117Z
base_sha: 73b4e99428db4a11b88721dc46a80f6e938f17e5
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk640
worktree: .worktrees/muse_attention_decode/full-decode1-cuda-splitkv-chunk640
mode: check
implemented_at: 2026-10-24T184625Z
impl_commit: 299b00d2433c741d41acc832edd2f24a7083a934
submission_sha256: f717653baf663168198a63c36f520a0fd43865d488be444851b892a50fdf5ce9
norm_sha256: d51a6f40c92eab33b483846edf12c405f4b2da2d3708bf4c7244ad7677b5e0f9
submission_blob: edf00ade1e0a0382d6beb3fdefaff5afade611f1
lint_warnings: 0
attempt_file: 2026-10-24T184625Z-check-full-decode1-cuda-splitkv-chunk640.md
prepared_at: 2026-10-24T184625Z
job: 91d541778b8f45f3
dispatched_at: 2026-10-24T184641Z
---

# full-decode1-cuda-splitkv-chunk640

## Change

Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's CUDA split-KV path instead of FA3. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e < CHUNK; e += THREADS)`); kernel bodies otherwise byte for byte. (2) splitkv_combine_kernel gets template <int CHUNK, int NW> (red arrays [NW], loop strides NW * 32 and NW, __launch_bounds__(NW * 32)). (3) Host functions take an int64 chunk argument (update _SPLITKV_CPP): 1024 -> <1024> attn and <1024, 4> combine (today's code); 640 -> <640> attn and <640, 8> combine; anything else TORCH_CHECK fails. Module renamed muse_full_decode_splitkv_v2. (4) Python: _splitkv_gate allows 1 <= batch; _full_decode_splitkv uses chunk = 640 if batch == 1 else 1024, max_chunks = ceil(page_table.shape[1] / chunk), grid unchanged (2 x SM count; at 80K keys only 125 items exist, so 125 CTAs get one 10-tile item each and the rest exit at the `bx >= total` check). (5) _splitkv_self_check adds batch-1 cases lens [80000] and [5000], kv_scale 1 and 0.25, grid device and 3, rel L2 < 2e-2. Print '[splitkv] batch-1 chunk 640 path ON/OFF' once at import.

## Expected

- `*layer=full load=decode_1 *` faster: incumbent 27.3 us (FA3 14.6 + combine 7.1 + pre 1.3 + norm_gate 1.1). 125 CTAs x 160 KB each with 3 tiles in flight per CTA; ~8-10 us for kernel A and only 125 partials for an 8-warp combine (~1.5-2.5 us): ~14-17 us shape
- `*layer=sliding*` flat: outside the gate
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: batch > 1 keeps chunk 1024
- `*load=verify_*` flat: outside the gate

## Risk

A/B sibling of full-decode1-cuda-splitkv-chunk320: 320 maximizes CTAs (250, ~2 per SM) at the cost of 250 partials; 640 halves the partials and keeps ~1 CTA per SM. The two differ only in the chunk/NW constants, so comparing their kernel A and combine rows attributes the trade. Untouched by every rewrite in the judge.

## Adversarial review

Reviewed 2026-10-24T184117Z by an adversarial subagent (model opus).

**Verdict: proceed.** Moving full decode_1 off FA3 (14.6 us attention + 7.1 us combine + 1.1 us norm-gate) onto the landed CUDA split-KV kernel is plausible and readable straight from the profile's per-kernel rows, and the sibling chunk320 is already the chunk-size A/B, so no companion is needed; the main implementation trap is the combine's four-warp reductions that are hardcoded for 4 warps and must become NW.

### Justification
Mechanism: full decode_1 is 27.3 us against a 4.3 us floor (80000 tokens x 256 B of K+V = 20.5 MB). 22.8 us of that is FA3 plus its combine plus rms_norm_gate. The landed split-KV kernel streams 2.94 TB/s at decode_16 (327 MB in 111.8 us), so a single-pass gather of 20.5 MB is physically 4-7 us, and the fused combine+norm-gate drops a launch. Shape math: 125 items of 640 tokens = 10 tiles of 64 tokens each, so SMEM_A is about 88 KB (a 64 KB ring, 5 KB idx, 16 KB acc), still 2 CTAs/SM; 125 CTAs cover 125 of 132 SMs in one wave. 640/256 is not an integer, which is why the strided issue_meta loop is required, and the hypothesis already says so. Expected directions match the gates: only full decode_1 changes. paired already excludes batch 1, so the q layout is unchanged; sliding, verify and batch>1 keep chunk 1024. Headline impact: the decode_1 load is about 46*13.2 + 16*27.3 = 1044 us. Saving 10-13 us x 16 gives a load ratio of about 1.18-1.25, which is about +2.8-3.8% on the 6-load geomean, just above the +/-1.5% CI. So judge it on the full decode_1 shape row and the kernel rows, not the headline. Weakest link: with one item per CTA there is no cross-item pipelining, so kernel A is latency-bound rather than bandwidth-bound. First the page-index meta (one dependent cold-L2 DRAM round trip) must land before any K/V address is known. Then 10 tiles stream with only 3 stages in flight (48 KB per SM), which is about 4 serialized DRAM latencies. 8-10 us for kernel A is plausible but could be optimistic; 10-12 us is a realistic downside that would still beat FA3+combine (21.7 us). The precedent loss (full-fused-triton-decode1-gate, -5% at 0.89 TB/s) was a different, Triton, kernel and is not a do-not-retry.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: this change templates its kernels, so the batch>1 path (decode_16 111.8 us, decode_32 245.9 us, both weight 16) must be byte-identical as the <1024> / <1024,4> instantiations. Any register or occupancy shift from templating would show up as movement on full decode_16/32, which the expected list says is flat. Check those rows: movement there is an implementation artifact, not the batch-1 effect.
- paired-heads-launch-cleanup-ab-tune-batch-gate: it is the reason batch 1 is currently unpaired FA3 with the judge's scheduler_metadata, i.e. the incumbent being replaced. This change bypasses it entirely at batch 1, so it confounds nothing beyond defining the baseline.
- fused-preattention-triton / paired-heads-launch-cleanup: they produce the unpaired FP8 q and write the cache. That part is unchanged at batch 1 (PAIRED=False already), so the 1.3 us pre row should be flat.
- sliding-deep-splits-on-cuda-splitkv / sliding-fused-triton-decode1-gate: no code interaction, but sliding decode_1 (weight 46) sits in the same load and has moved between runs before (full-decode-cuda-gather-splitkv 'lost on sliding decode_1' and fp8-qk-mma 'gained' on it with no sliding code change). Its noise can mask or inflate the load-level change, so attribute using the full decode_1 shape row.

### Interference and A/B needs
No companion is warranted. A gain or a loss on full decode_1 is attributable from the profile alone, because the change swaps the FA3 + FlashAttnFwdCombine + rms_norm_gate rows for splitkv_attn_kernel<640> + splitkv_combine_kernel<640,8> rows, each of which reads directly. No landed hypothesis acts on this shape's attention any more, so an ab-revert has nothing to isolate. The obvious ab-tune (chunk size, and with it CTA count versus partial count) is already queued as the sibling full-decode1-cuda-splitkv-chunk320, which differs only in the CHUNK/NW constants. A control that templates the code but keeps batch 1 on FA3 would only re-measure the batch>1 shapes, and those rows already show whether templating is neutral. Run order: 320 doubles the bytes in flight per SM (2 CTAs/SM x 3 stages) for a kernel that is ramp/latency-bound at one item per CTA, so it is the more likely winner on kernel A. 640 is the safer one on the combine. Either order works, and the second run settles the chunk trade.

### Open questions
- splitkv_combine_kernel hardcodes 4 warps in more places than the change lists: M = fmaxf(fmaxf(red_m[0], red_m[1]), fmaxf(red_m[2], red_m[3])), the first-pass stride c += 128, the c += 4 warp stride, and the warp-0 final loop for (w = 0; w < 4; ++w). All of these must become NW-generic. Missing the red_m reduction gives a too-small M and possible exp2 overflow (the self-check's finite/L2 test should catch it, but it would silently disable the whole split-KV path, batch>1 included, via the SELF-CHECK FAILED fallback, which would look like a big regression on decode_16/32).
- If the self-check fails, _EXT = None also turns off the landed batch>1 path. Consider whether a batch-1-only failure should disable only the batch-1 gate, so one bad run does not cost the decode_16/32 gains and muddle attribution.
- With only 125 CTAs on 132 SMs and one item each, does the block scheduler place one CTA per SM (as assumed), or does it pack 2 per SM on fewer SMs? The latter would halve per-SM bandwidth headroom; check splitkv_attn_kernel<640> duration against the ~8-10 us estimate.
- Is the 7.1 us FA3 combine at batch 1 partly launch-gap/serialization that the new 2-kernel sequence also pays? The gap between kernel A's end and the combine's start at batch 1 is worth reading in the profile, as the follow-up (PDL attn->combine, already queued as splitkv-pdl-attn-to-combine) depends on it.


## Implementation notes

Only `muse_attention_decode/submission.py` in the worktree changed (+108/-61).
- CUDA: `CHUNK`/`TILES` global constants removed; `splitkv_attn_kernel<int CHUNK>` derives TILES and the smem offsets from a `Lay<CHUNK>` struct (SMEM_A per instantiation); `issue_meta` loops `for (e = tid; e < CHUNK; e += THREADS)`. Bodies otherwise unchanged, so `<1024>` is today's kernel.
- `splitkv_combine_kernel<CHUNK, NW>`: red arrays [NW], first-pass stride NW*32, warp stride NW, warp-0 final loop NW, and the red_m max reduction is now a loop over NW (the review's trap), `__launch_bounds__(NW*32)`, launched with NW*32 threads.
- Host: `launch_attn<CHUNK>` / `launch_combine<CHUNK,NW>` templates (per-instantiation cudaFuncSetAttribute flag); `splitkv_attn`/`splitkv_combine` take int64 chunk (1024 -> <1024>/<1024,4>, 640 -> <640>/<640,8>, else TORCH_CHECK). `_SPLITKV_CPP` updated; module `muse_full_decode_splitkv_v2`.
- Python: gate allows batch 1 only when `_B1_OK`; chunk = 640 at batch 1 else 1024; max_chunks from that chunk; grid unchanged.
- Deviation (per review open question 2): batch-1 self-check (lens [80000] and [5000], kv_scale 1 and 0.25, device and 3-CTA grids, rel L2 < 2e-2, finite) runs after the batch>1 check, and a failure there only sets `_B1_OK = False` (batch 1 stays on FA3, batch>1 split-KV stays on). Prints `[splitkv] batch-1 chunk 640 path ON/OFF`. Gate never raises. Not run (no GPU).

## Rules check

- [x] CUDA-graph capturable: no .item()/.cpu()/synchronize() in kernel_fn; the only new host work is shape/config arithmetic; the self-check is import-time only.
- [x] Nothing derived from `data` is cached across calls (only a per-instantiation static attribute flag; `_B1_OK` is set from synthetic data at import, not from inputs).
- [x] Weights: n/a: no weights touched.
- [x] Activations/KV/query dtypes unchanged (FP8 KV, FP8 e4m3 q, bf16 out).
- [x] Attention sees every key: positions < cache_seqlens all processed (chunk 640 items cover ceil(len/640) chunks); no skipping.
- [x] No timing patches, harness access, reference import, subprocess or network (diff checked).
- [x] All work inside the timed call; no output caching or replay.

okc_lint output: `CLEAN: submission.py`

## Pre-flight audit

Audited 2026-10-24 (opus, read-only plus emulation; no GPU). **Verdict: dispatch-with-fixes.**

- Compile: templated kernels and host launchers read clean (Lay<CHUNK> static constexpr offsets, `splitkv_attn_kernel<CHUNK>` passed to cudaFuncSetAttribute, `__launch_bounds__(NW * 32)`, templates inside the anonymous namespace with public entry points outside). The `#pragma unroll` on the now tid-strided meta loop can at most produce a "not unrolled" warning, not an error.
- issue_meta coverage at CHUNK 640: emulated `for (e = tid; e < 640; e += 256)` over tid 0..255; it covers 0..639 exactly once each (tids 0..127 take 3 iterations, the rest 2). TILES = 10 ≥ STAGES - 1, so the meta double buffer (items k-1 and k+1 sharing a buffer) is still safe.
- Shared memory: SMEM_A is 90,128 B at <640> and 93,200 B at <1024>. Two CTAs per SM need 182-188 KB, under the H200 228 KB per SM and 227 KB per block. Combine<640,8> static smem is about 2.1 KB. Offsets stay 16-byte aligned for the cp.async16 and float4 accesses.
- NW=8 combine: every hardcoded 4 is generalized (red arrays [NW], first-pass stride NW*32, the red_m max is a loop over all NW warps, warp stride NW, final loop NW). Warps with no partials contribute M=-inf, L=0 and acc=0. Fine for n < NW.
- Batch-1 self-check: the kernel and the reference consume the same FP8 q and FP8 K/V, so the residual error is fp16 P and fp32 accumulation, around 1e-3 against a 2e-2 bound. Items are ceil(80001/640) = 126 ≤ max_chunks = ceil(80038/640) = 126. On a miss, `_B1_OK = False` returns only batch 1 to FA3 and the check never raises (it catches exceptions; `_SPLITKV_GRID` is reset in finally).
- Fix: added one `[splitkv] PATH CHOSEN: ...` import-time line after the build and self-check block. It reports the batch>1 and batch-1 paths on every outcome: build fail, batch>1 miss, batch-1 miss or all ON. Reason: the spec ON/OFF line is skipped when the build or the batch>1 check fails. Lint CLEAN, py_compile OK.
- Rules: no new host sync, allocation or input-dependent control flow in kernel_fn (the chunk is chosen from config batch). Nothing data-derived is cached. Dtypes are unchanged. Every key is attended.

Residual risk: a sticky CUDA fault (rather than a numeric miss) in the never-run <640>/<640,8> instantiations during the batch-1 self-check would poison the context and cannot fall back. Also, templating may shift <1024> codegen slightly; watch the full decode_16/32 rows.
