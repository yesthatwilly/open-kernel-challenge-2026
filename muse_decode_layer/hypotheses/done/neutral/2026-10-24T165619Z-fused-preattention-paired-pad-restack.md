---
problem: muse_decode_layer
slug: fused-preattention-paired-pad-restack
created: 2026-10-24T165619Z
base: master
area: pre-attention norm+rope
change: >
  Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch
  muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch, _pre_attention_kernel
  reading RoPE from state['cos_sin']) onto master once paired-heads-pad-unpad-triton has landed
  (build on master if it landed, else on its branch), and finish the port of
  muse_attention_decode's landed chain fused-preattention-triton + paired-heads-launch-cleanup
  part (1) 'pad at the source', which decode_layer never got. 3-way merge the patch's kernel, its
  prepare() warm-up and its kernel_fn call onto master, then add attention master's PAIRED and G
  constexprs to the kernel verbatim (muse_attention_decode/submission.py on master,
  _pre_attention_kernel lines 117-124: when PAIRED, query program h writes its 64 FP8 values into
  half (h // G) % 2 of a [tokens, Hq, 128] FP8 row and 64 FP8 zeros into the other half). In
  kernel_fn: one launch, grid (tokens, Hq + Hkv), num_warps 1, replaces state['qk_norm'](...), the
  k/v divide by the scales, k.to(FP8), v.to(FP8), store_cache(...) and either
  q.view(T,16,64).to(FP8) (unpaired) or the _pair_pad_q_kernel launch that
  paired-heads-pad-unpad-triton added (paired: window == 0 and batch > 1 and hkv % 2 == 0,
  unchanged predicate, passed as PAIRED). q buffer is torch.empty((tokens, hq, 2*d if paired else
  d), FP8). gate and v stay slices of qkv; FA3, _paired_norm_gate_kernel / o_norm_gate and
  everything after the o GEMM are byte-identical to the base. If sliding-fused-triton-decode1-port
  has landed, its sliding batch-1 decode branch keeps running before this code and is untouched;
  if full-decode-cuda-splitkv-port has landed, pass PAIRED=False on its gated shapes exactly as
  attention master does (paired = ... and not splitkv). Delete _pair_pad_q_kernel only if nothing
  else calls it.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: FusedQKNorm + 2 FP8 casts + store_cache + _pair_pad_q_kernel (5 launches) become 1; attention saved 3.5-6.4 us for this 5->1 on 17-36 us shapes, ~1% of ~400 us
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: same 5->1 at T=128 on the paired path, ~4-6 us of ~580 us
  - shapes: "*load=decode_1*"
    direction: faster
    why: unpaired 5->1 on latency-bound 167-177 us shapes; the neutral parent moved these +2.3% (full decode_1 also loses its q cast); sliding decode_1 flat if sliding-fused-triton-decode1-port landed (it bypasses this code)
  - shapes: "*load=verify_1*"
    direction: faster
    why: T=8 unpaired 5->1 on ~230-245 us shapes, ~1-2%
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: the parent gained 3.3% here, the one shape outside drift
  - shapes: "*load=verify_32*"
    direction: flat
    why: few us on ~620-910 us shapes; Marlin dominates
risk: low
depends_on: paired-heads-pad-unpad-triton
priority: 5
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T170604Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T034153Z
base_sha: 8ac7a2f409615586e09a8a8ce118f9c4f892b7ba
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/fused-preattention-paired-pad-restack
worktree: .worktrees/muse_decode_layer/fused-preattention-paired-pad-restack
mode: check
restack_of: paired-heads-pad-unpad-triton
stale_base: null
implemented_at: 2026-10-25T034605Z
impl_commit: 963bcf02677e391bcb80d9d64971b5047bfcc4c8
submission_sha256: 2b321f331e1f7c8cdd1fa4dd6acac23425374816ef0eeb8479ef672de2427188
norm_sha256: 6a5ef1194cced186c0e9ede6711aafde54eba56d230f8edda917b488c2b28b80
submission_blob: 549a90e07960194c47c7589e9fd5c35a5061cb0a
lint_warnings: 28
attempt_file: 2026-10-25T034605Z-check-fused-preattention-paired-pad-restack.md
prepared_at: 2026-10-25T034605Z
job: 5e9b5464d1444708
dispatched_at: 2026-10-25T034613Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2744529931842001
speedup_ci: [1.255336198286437, 1.293569788081963]
harvested_at: 2026-10-25T043358Z
best_at_harvest: 1.2680313318618024
promotion: null
---

# fused-preattention-paired-pad-restack

## Change

Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch, _pre_attention_kernel reading RoPE from state['cos_sin']) onto master once paired-heads-pad-unpad-triton has landed (build on master if it landed, else on its branch), and finish the port of muse_attention_decode's landed chain fused-preattention-triton + paired-heads-launch-cleanup part (1) 'pad at the source', which decode_layer never got. 3-way merge the patch's kernel, its prepare() warm-up and its kernel_fn call onto master, then add attention master's PAIRED and G constexprs to the kernel verbatim (muse_attention_decode/submission.py on master, _pre_attention_kernel lines 117-124: when PAIRED, query program h writes its 64 FP8 values into half (h // G) % 2 of a [tokens, Hq, 128] FP8 row and 64 FP8 zeros into the other half). In kernel_fn: one launch, grid (tokens, Hq + Hkv), num_warps 1, replaces state['qk_norm'](...), the k/v divide by the scales, k.to(FP8), v.to(FP8), store_cache(...) and either q.view(T,16,64).to(FP8) (unpaired) or the _pair_pad_q_kernel launch that paired-heads-pad-unpad-triton added (paired: window == 0 and batch > 1 and hkv % 2 == 0, unchanged predicate, passed as PAIRED). q buffer is torch.empty((tokens, hq, 2*d if paired else d), FP8). gate and v stay slices of qkv; FA3, _paired_norm_gate_kernel / o_norm_gate and everything after the o GEMM are byte-identical to the base. If sliding-fused-triton-decode1-port has landed, its sliding batch-1 decode branch keeps running before this code and is untouched; if full-decode-cuda-splitkv-port has landed, pass PAIRED=False on its gated shapes exactly as attention master does (paired = ... and not splitkv). Delete _pair_pad_q_kernel only if nothing else calls it.

## Expected

- `*layer=full load=decode_16*` faster: FusedQKNorm + 2 FP8 casts + store_cache + _pair_pad_q_kernel (5 launches) become 1; attention saved 3.5-6.4 us for this 5->1 on 17-36 us shapes, ~1% of ~400 us
- `*layer=full load=verify_16*` faster: same 5->1 at T=128 on the paired path, ~4-6 us of ~580 us
- `*load=decode_1*` faster: unpaired 5->1 on latency-bound 167-177 us shapes; the neutral parent moved these +2.3% (full decode_1 also loses its q cast); sliding decode_1 flat if sliding-fused-triton-decode1-port landed (it bypasses this code)
- `*load=verify_1*` faster: T=8 unpaired 5->1 on ~230-245 us shapes, ~1-2%
- `*layer=sliding load=decode_32*` faster: the parent gained 3.3% here, the one shape outside drift
- `*load=verify_32*` flat: few us on ~620-910 us shapes; Marlin dominates

## Risk

Low expected value (~+1% headline, near the 1.5% drift), so judge it per shape: the paired rows against paired-heads-pad-unpad-triton's rows, the unpaired batch-1 rows against the neutral parent's. It is the only attention-side landed piece not yet ported or queued for decode_layer (port audit 2026-10-24). Confounders: full-decode-cuda-splitkv-port replaces the paired path on full decode_16/32 if it lands first (then only verify_16/32 keep the paired pad), and sliding-fused-triton-decode1-port removes sliding decode_1 from this code path. Correctness: the k/v cache store moves into Triton, so the cache contents are checked by the judge's later reads; keep the parent's kv_scale handling (divide before the cast) and cache_loc int64 indexing.

## Adversarial review

Reviewed 2026-10-24T170604Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean restack of decode's own neutral fused-preattention-triton onto the current master (1.1665x). It adds attention master's PAIRED in-kernel pad, so 5 launches become 1 on every shape. Expect about +0.8-1.2% on the headline, inside the CI, so judge it per shape on the batch-1 rows. Dispatch it only after paired-heads-pad-unpad-triton (job a92b64be4c6547d3) has harvested, and fix the conditional base before claim time. No companion is needed.

### Justification
Mechanism: on master kernel_fn the pre-attention step is still SGLang's FusedQKNorm, the two k/v .to(FP8) casts, the store_cache JIT and either q.view().to(FP8) (unpaired) or, once paired-heads lands, _pair_pad_q_kernel. That is five launches of a few hundred bytes to a few hundred KB each, all latency-bound. One Triton program per (token, head) at num_warps 1 is roughly 2-3 us at T<=32, so about 4-7 us is saved per shape. The same 5->1 gave muse_attention_decode 3.5-6.4 us. Profile support is indirect. In master's profile (9dcaec316b3045a7) none of the five kernels reaches the top-12 rows on any shape; they are all in the '... N more' tail (each under about 2.8 us at full decode_1). The archived profile text and JSON carry only the truncated report, so the replaced kernels' own us cannot be read from top_kernels; only the row count ('... 14 more' vs '... 21 more') and the shape totals can. The parent's per-shape rows are the best evidence: decode_1 +1.6-2.1% (sliding 172.8->167.0, full 182.1->176.8), verify_1 +0.2-1.7%, sliding decode_32 +3.3%, and everything else flat or within -1%, including full verify_32 at T=256 with 4,608 programs (0.990, no loss). Since then post-expert-norm-triton-small-t has shrunk the T<32 totals (sliding decode_1 is now 145.8 us, full 157.0 us), so the same ~4-5 us is now about 3% of those shapes. That makes the decode_1 'faster' and verify_1 'faster' directions credible. The sliding decode_32 'faster' is weakly founded: it was the parent's one out-of-drift shape, at 3.3% when the replaced launches are worth about 1.5% of 252 us there. Treat it as probably drift rather than a prediction. Full decode_16 and verify_16 'faster' on the paired path is about 1% of 365/604 us, which is inside the per-shape drift. Verify_32 'flat' is right. On the floors, every shape sits 3-4x off speed of light, dominated by Marlin, FA3 and the nvjet GEMMs, so this is launch hygiene and not a move toward SoL. Headline arithmetic: decode_1 load about +2.5-3%, verify_1 about +1.5%, the other four loads about +0.3-0.7% each, so the geomean is about +0.8-1.2%. That is below the ~1.5% CI half-width, and the result will very likely classify neutral or within-ci again. It needs promotion by hand on per-shape evidence, as the DECISIONS precedent did. Weakest link: the expected magnitude rests on launch-count arithmetic for kernels this profile cannot show. The parent's own decode_1 gain (+2%) is the same size as the drift LESSONS documents (1.5-2.5% per shape), so even the best-evidenced rows can be read either way from one run. A second, smaller weakness is that the spec tells the implementer to merge the parent's 'prepare() warm-up', but the parent patch has no warm-up; its prepare() hunk only removes qk_norm.

### Confounders on the current kernel
- paired-heads-pad-unpad-triton (running, depends_on, not yet landed): the spec's base is conditional ('master if it landed, else on its branch'). If this is built on the unmerged branch, the measured delta against BEST mixes that hypothesis's own pad/unpad effect into this one on all four full batch>1 shapes. If paired-heads ends neutral and is not merged, there is no defined base at all. Fix the base at claim time: build on master after that harvest. If paired-heads did not merge, PAIRED replaces master's torch.zeros plus the two slice copies (3 launches) and the un-pad copies plus FusedRMSNormGate stay, and the paired rows are then read against paired-heads' own rows, not against BEST.
- fa3-hdim128-head-pairing-batched (landed): the PAIRED write only exists because of the paired path. This is an interaction by construction, not a confounder. Reverting it would delete the PAIRED half of the change.
- post-expert-norm-triton-small-t (landed): it shares no code or buffers with the pre-attention step, but it cut the T<32 totals by 10-15%, which raises this change's percentage on decode_1/verify_1 relative to the parent's run. Compare per-shape us, not the parent's percentages.
- full-decode-cuda-splitkv-port and sliding-fused-triton-decode1-port (running, not landed): either can remove shapes from this code path (full decode_16/32 and sliding decode_1). The splitkv branch keeps FusedQKNorm, the casts and store_cache, and takes a contiguous FP8 q [T,16,64]. This kernel with PAIRED=False produces exactly that, so it is compatible, but the restack must route that branch through the fused kernel too, because state['qk_norm'] is deleted. The implementer must read the actual master at claim time, not the spec's if/else.
- marlin-block-m-cap-16 and sglang-pipeline-parity-router-hilo-m8 (landed): different areas with no shared launches. They contribute only seed-dependent Marlin drift on the totals; the per-case seeds differ between runs.

### Interference and A/B needs
No companion is warranted. An ab-revert of fa3-hdim128-head-pairing-batched removes the code under test. An ab-revert of post-expert-norm or the Marlin cap would only change the denominators. A control that isolates the PAIRED sub-change (in-kernel pad vs _pair_pad_q_kernel) would measure one ~2 us launch on the 16-weight full batch>1 shapes, which is far below drift, so it could not disambiguate anything. The useful split is already in the run's own shapes. The unpaired batch-1 rows (full/sliding decode_1 and verify_1) measure the 5->1 fusion without PAIRED. The paired full batch>1 rows measure it with PAIRED. Full verify_32 at T=256 is the regression canary. A loss would be attributable without a companion: it would show at full/sliding verify_32, or as a correctness failure in the cache store. A gain is ambiguous only statistically, and a companion run would not fix that. If quota is tight, fold this into a combined decode check (for example with moe-align-small-t-restack-on-master, which is also inside CI) rather than spending a scored submit on about +1%.

### Open questions
- Which base at claim time? Gate dispatch on paired-heads-pad-unpad-triton's harvest (job a92b64be4c6547d3) rather than building on its unmerged branch. If it is not merged, should PAIRED replace master's torch.zeros plus the two copies while the un-pad block is left alone? That is the only definition that keeps this hypothesis's scope.
- The spec says to merge the parent's 'prepare() warm-up', but the parent patch has none. Confirm that the implementer should skip it.
- The truncated profile report (top 12 plus '... N more') cannot show the replaced kernels. Can the harvest record the full kernel list, or at least the '... N more' count and total per shape, so the 5->1 can be confirmed as launches actually removed rather than inferred from shape totals?
- With an expected headline of +0.8-1.2% inside the CI, is a scored submit worth it, or should this go into a combined decode check with the other inside-CI restacks (moe-align-small-t-restack-on-master)?
- If sliding-fused-triton-decode1-port lands first, sliding decode_1 (weight 46, this change's best-evidenced shape) leaves this code path, and the remaining expected gain drops to about +0.6%. Re-estimate the priority at that point.


## Implementation notes

Worktree base was clean master (8ac7a2f), because the restack's 3-way apply conflicted. Only `muse_decode_layer/submission.py` changed (+129/-36).

Merge resolution:
1. Hand-merged the parent paired-heads-pad-unpad-triton.patch onto current master. Master already carried the torch.zeros/slice-copy paired path (the parent's target text had moved), so I swapped those blocks for the patch's `_pair_pad_q_kernel` launch and its `_paired_norm_gate_kernel` un-pad+norm+gate launch. The un-paired branch keeps `state["o_norm_gate"]`. Both kernels are placed just before `_sliding_forward`, since master's `prepare()` now sits after the sliding and dense-chain code. The `prepare()` hunk had only context changes, so nothing was needed there.
2. Applied this hypothesis: `_pre_attention_kernel` (from fused-preattention-triton.patch) with attention master's `G` and `PAIRED` constexprs, taken verbatim. It replaces `_pair_pad_q_kernel`, which was deleted because nothing else calls it. In `kernel_fn`: one launch on grid (tokens, hq+hkv), num_warps 1. It replaces `qk_norm`, the k/v scale divides, both FP8 casts, `store_cache` and the pair pad. The q buffer is `torch.empty((tokens, hq, 2*d if paired else d), FP8)`. `paired = window == 0 and batch > 1 and hkv % 2 == 0` is unchanged and is passed as PAIRED. The FA3 call, the descale and `_paired_norm_gate_kernel` are untouched.
3. Removed the `qk_norm` entry from `prepare()`, plus the now-unused `store_cache` and `FusedQKNorm` imports. Updated the module docstring line.
4. Kept intact: the sliding batch-1 `_sliding_forward` branch (it runs before this code), the small-t dense chain with PDL, and the align-par/side-stream code. full-decode-cuda-splitkv-port has not landed on master, so there is no splitkv gate and PAIRED is not forced to False anywhere.

Deviations: the parent fused patch has no prepare() warm-up (only the qk_norm removal), so none was merged. `v` is no longer a slice in `kernel_fn`, since the kernel reads it from qkv; `gate` is still a slice of qkv. Nothing could be run on this node: no GPU, torch or Triton.

## Rules check

- [x] CUDA-graph capturable: `git diff` adds only `torch.empty` and Triton launches in `kernel_fn`, with no `.item()`, `.cpu()` or sync, and no control flow on values. The `paired` and `scaled` branches read config only.
- [x] Nothing derived from `data` is cached: the q buffer is allocated per call, and cos_sin is a prepare() table indexed by the positions input.
- [x] Weights: n/a, no weight change; only `qk_norm` was removed from state, and it held no weights.
- [x] Activations and KV: the cache stays FP8, divided by the scales before the cast as before. The query is FP8 E4M3 only, as on master (master also cast q to FP8).
- [x] Attention sees every key: FA3 call and window are unchanged, and the pad half is zero, as on master.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: the diff adds none.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune; the first-line marker is kept.

okc_lint: `OK with warnings` (0 errors). The warnings are pre-existing host-sync lines in prepare()/self-check code, outside kernel_fn.
py_compile: ok; `__pycache__` removed.

## Pre-flight audit

Verdict: **dispatch** (no fixes).

- Deleted dependencies: `qk_norm`, `store_cache`, `FusedQKNorm` and `_pair_pad_q_kernel` appear only in comments/docstrings. `_sliding_forward` has its own in-kernel norm/RoPE/store, and the dense-chain PDL, align-par and prepare() self-checks never referenced them, so there is no NameError path.
- Diff against master is confined to the module docstring, the two imports, the two new kernels (inserted before `_sliding_forward`), the `qk_norm` state entry and the non-sliding-decode-1 attention block of kernel_fn. The landed stack-align-par-sidestream-sliding1 and small-t-launch-minimal-dense-chain-pdl code is byte-identical.
- Numerics: the fp32 scaleless RMSNorm (eps inside the mean), then interleaved RoPE from the same cos | sin table (cols 0..31 cos, 32..63 sin; even = e*c - o*s, odd = e*s + o*c), matches reference `_rope` and master's `_norm_rope`. The q E4M3 cast now comes straight from fp32, which matches the reference's FP8-query path more closely than master's bf16 -> E4M3. The k/k_scale and v/v_scale fp32 -> E4M3 casts equal reference `_quant`, compared at CACHE_TOL rather than as bytes. The cache index is `(loc*HKV + kh)*D` on the contiguous [pages, page, 2, 64] FP8 cache, the same flat row as store_cache's view(-1, hkv*d). The v offset `(HQ+HKV+kh)*D` and the gate offset `(HQ+2HKV)*D + h*D` match the qkv split.
- Paired layout: query h writes half (h//G)%2 of its [2D] row and zero-fills the other half. That is identical to master's `view(tokens, s, 2, g, 2d)` copy. `_paired_norm_gate_kernel` reads the same half from FA3's [T, HQ, 2D] output through its real strides.
- Rules: kernel_fn adds only `torch.empty` and two Triton launches; PAIRED/ROPE/scaled are config-only; nothing cached from data; no sync, autotune or child process. FP8 tensors passed directly to Triton (q, k/v cache) were already exercised by the parent patch's q pad. Lint OK (pre-existing prepare() warnings only), py_compile OK.

Residual risk: Triton FP8 casts saturate where torch's produce NaN, and q/k now skip master's intermediate bf16 rounding. Both stay within tolerance at realistic magnitudes, but the run is the first execution of _pre_attention_kernel on the full and sliding batch>1/verify shapes.

## Result

- job `5e9b5464d1444708` (check), status done, passed True, tests 13/13
- speedup 1.2745x, 95% CI [1.2553, 1.2936], score 17749.02653999309 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/fused-preattention-paired-pad-restack.result.json`, `muse_decode_layer/hypotheses/done/neutral/fused-preattention-paired-pad-restack.patch`; branch `hyp/muse_decode_layer/fused-preattention-paired-pad-restack`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 127.48 | 125.43 | 1.0148 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.93 | 138.37 | 1.0323 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 204.98 | 211.33 | 1.0134 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 349.47 | 336.11 | 1.0467 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 245.77 | 229.71 | 1.0282 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 529.71 | 512.59 | 1.0326 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 190.8 | 182.44 | 1.0374 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 192.44 | 192.51 | 1.0158 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 429.61 | 435.24 | 0.9963 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 563.65 | 563.83 | 1.0059 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 592.52 | 619.5 | 1.0101 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 861.3 | 868.91 | 1.0197 | flat |

gained on full decode_1, decode_16, decode_32; sliding verify_1. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: fused-preattention-paired-pad-restack`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

