# muse_attention_decode · 2026-10-24T124759Z · check · fa3-scheduler-metadata-side-stream

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T120457Z-fa3-scheduler-metadata-side-stream.md` (branch `hyp/muse_attention_decode/fa3-scheduler-metadata-side-stream`).

- **Job:** `42964d938c5842f4` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `83e8e87`

## Goal
Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash::prepare_varlen_num_blocks_kernel inside the call, serially after _pre_attention_kernel (1.7-1.8 us on sliding, 2.8-3.6 us on full; see the batch-gate profile bd64bf9f9e904f02). That kernel reads only cache_seqlens / cu_seqlens_q and shapes, not anything the pre-attention kernel writes, so build the same schedule concurrently. In kernel_fn, before the _pre_attention_kernel launch: cur = torch.cuda.current_stream(); _SIDE (one module-level torch.cuda.Stream() created lazily per device, never derived from data).wait_stream(cur); under torch.cuda.stream(_SIDE): meta = sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=L, max_seqlen_k=page_table.shape[1]*page_size, num_heads=Hq, num_heads_k=attn_heads (Hkv unpaired, Hkv/2 paired), headdim=2*d if paired else d, headdim_v same, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, window_size=(window-1, 0) if window else (-1, -1), num_splits=0, pack_gqa=None, sm_margin=0), i.e. exactly the arguments flash_attn_with_kvcache passes to its internal scheduler for this call (check the installed sgl_kernel flash_attn.py signature and mirror its defaults; if any argument such as max_seqlen_k or has_softcap differs, use what the in-call path derives). Then launch _pre_attention_kernel on cur as now, cur.wait_stream(_SIDE), and pass scheduler_metadata=meta to FA3. Also record meta on cur (meta.record_stream(cur)) so the allocator does not reuse it. Full decode_1 keeps the judge-built metadata unchanged (it already has no prepare kernel in its profile). Nothing else changes: query layout, pairing gate, epilogues, descale. This differs from the held fa3-scheduler-metadata-full-verify: that one launched the same scheduler serially (moving cost, not removing it) and used the unpaired geometry; this one runs it on a forked graph branch in parallel with the pre-attention kernel and uses the geometry of the actual call.

## What changed, and where
Built on master 10d039a (untrimmed sliding page table; fa3-window-trimmed-page-table regressed, so nothing trimmed). Only `muse_attention_decode/submission.py` changed, only around the FA3 call in `kernel_fn` (plus imports and two module-level helpers).

- Imports `get_scheduler_metadata` next to `flash_attn_with_kvcache`, and `inspect`.
- Module level: `_SIDE` dict plus `_side_stream(device)`, which creates one `torch.cuda.Stream` per device on first use. The first use is the first eager test call, before any capture. It is never per call and never derived from data.
- Module level: `_SCHED_WINDOW = "window_size" in inspect.signature(get_scheduler_metadata).parameters`, printed once at import (`[okc] get_scheduler_metadata accepts window_size: ...`), so the run log answers the review's open question. **Deviation from the brief:** the brief says to check that the wrapper accepts `window_size` and otherwise stop and report. No sgl_kernel source or install exists on this node (searched the filesystem), so I could not check statically. Instead the code does what the review's open-question guidance says to do in the "no" case: if the wrapper lacks `window_size`, sliding calls keep `metadata=None` (in-call prepare, as on master), and only the full-layer half (paired batch>1 and verify_1) uses the side-stream schedule. Read the log line on the run. If it prints False, the sliding expectations do not apply.
- The FA3 geometry block (paired views, `attn_heads`, judge metadata) now runs before the `_pre_attention_kernel` launch. It is views only, no launches. `window_size` is computed once and passed to both FA3 and the scheduler.
- `own_schedule = (paired or window != 0 or length != 1) and (window == 0 or _SCHED_WINDOW)`. This is config-only: every call that master made with metadata None. Full unpaired decode (decode_1 full) keeps `data["scheduler_metadata"]` unchanged.
- Fork, then side launch, then pre-attention, then join: `side.wait_stream(cur)` comes first. Then `get_scheduler_metadata` runs under `torch.cuda.stream(side)`, `_pre_attention_kernel` launches on `cur` as before, and `cur.wait_stream(side)` plus `metadata.record_stream(cur)` happen before `flash_attn_with_kvcache`.
- Scheduler arguments come from the tensors actually passed to FA3, never from cfg alone:
  - `batch_size=cache_seqlens.shape[0]`, `max_seqlen_q=L`
  - `max_seqlen_k=page_table.shape[1] * k_attn.shape[1]`, the shape-derived value mha_fwd uses for paged KV, with no host sync
  - `num_heads=q_attn.shape[1]`, `num_heads_k=k_attn.shape[2]` (Hkv unpaired, Hkv/2 paired), `headdim=q_attn.shape[-1]` (2d paired, d unpaired)
  - `cache_seqlens`, `qkv_dtype=q_attn.dtype` (FP8), `cu_seqlens_q`, `page_size=k_attn.shape[1]` (an explicit int, 1 in the scored cases), `causal=True`, `num_splits=0`
- `headdim_v`, `has_softcap`, `pack_gqa` and `sm_margin` are left at the wrapper defaults (headdim, False, None, 0). Those match the FA3 call. I left them out so that a wrapper naming them differently does not fail. Only the keywords the reference itself uses, plus `causal`/`window_size`, are passed.
- Window normalisation, following the review: before the scheduler call, apply mha_fwd's `left >= max_seqlen_k-1 -> -1` and `right >= max_seqlen_q-1 -> -1` clamps in Python, then pass `causal=True`. The C++ scheduler's own `seqlen_q==1 && both -1 -> non-causal` and `causal -> right=0` steps then reproduce mha_fwd. Examples: sliding at large context becomes (2047,0) local. The ctx-10 page-64 test gets (-1,-1), which at L=1 is non-causal as in mha_fwd. Full verify gets (-1,0) causal.

Uncertainties:
1. Not verifiable here: whether the installed scheduler's L==1 causal collapse matches mha_fwd's hdim-128 paged-KV exception (newer FA3 keeps causal for hdim 128 + paged at L=1). This matters only for paired full decode_16/32. The review's profile test covers it: the FA3 main/combine rows must stay within ~1% of bd64bf9f9e904f02. If they move, the schedule differs.
2. Whether the wrapper's keyword names match beyond those the reference uses (`causal`, `window_size`). `causal` is a standard parameter, and `window_size` is guarded.
3. Whether a TypeError would surface. If the wrapper rejects a keyword, the first test fails at once with a clear TypeError, not silently.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/fa3-scheduler-metadata-side-stream diff` (1 file, +61/-11).

- [x] CUDA-graph capturable. There is no `.item()`, `.cpu()`, `synchronize()` or tensor print in kernel_fn, and `max_seqlen_k` comes from shapes, not `cache_seqlens.max()`. The only branch is `own_schedule`, from config/shapes plus an import-time library signature check. The fork (`side.wait_stream(cur)`) comes before the side launch, and the join (`cur.wait_stream(side)`) before FA3, so the capture has no unjoined branch. The stream is created on the first eager call, not during capture. The import-time `print` is a string, not a tensor.
- [x] Nothing derived from `data` is cached across calls. `_SIDE` holds only a Stream per device. The schedule is rebuilt every call, and `metadata` is a local.
- [x] n/a: weights. This problem has no weights or `prepare()`, and no repacking was added.
- [x] Activations BF16 where SGLang keeps BF16, the KV cache FP8, queries FP8 E4M3. The query layout, cache writes, FA3 call and epilogues are unchanged. The scheduler takes `qkv_dtype=q_attn.dtype` (FP8), which matches the call.
- [x] Attention sees every key. The FA3 `window_size` and `causal` are unchanged (the same expression, now a local). Only FA3's tile schedule is supplied explicitly, with no top-k, sparse or block-skip.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new imports are `inspect` (only `inspect.signature` on an sgl_kernel function, which reads no frames) and `get_scheduler_metadata` from sgl_kernel.
- [x] All work happens inside the timed call, with no output caching or replay. The schedule is computed inside kernel_fn on a captured side branch. Fork-first keeps it from running eagerly outside the graph.
- [x] The first-line marker `# okc: problem=muse_attention_decode` is kept. There is no `triton.autotune`.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/fa3-scheduler-metadata-side-stream/muse_attention_decode/submission.py
```

`python3 -I -m py_compile` passed, and the `__pycache__` it left was removed from the worktree.

## Risks / what could fail
medium
Correctness risk: a schedule built for a different geometry than the call (headdim, num_heads_k, window, page_size, max_seqlen_k) can mis-size splits; the reviewer should confirm against sgl_kernel flash_api mha_fwd how it derives max_seqlen_k and pack_gqa for paged KV and that a passed metadata tensor fully replaces the in-call prepare (full decode_1 profile shows no prepare kernel when metadata is passed). Perf risk: a graph fork/join adds edges, not events, but if FA3 PDL overlap of prepare with pre-attention already hides part of it (GPU busy rows are ~1 us under the kernel sums) the gain shrinks to under 1 us. Not input-derived caching: the schedule is rebuilt every call; only the stream object is module-level. If the running fa3-window-trimmed-page-table lands, the sliding metadata must use its trimmed page table width and lengths.

## Result (after the run)
- job `42964d938c5842f4` (check), status done, passed True, tests 16/16
- speedup 1.2245x, 95% CI [1.2062, 1.2466], score 3075.6828334114957 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2930643047593426, "ci": [1.2736683401879525, 1.3211607975952195], "job": "295727fc68e14093"}; base result: muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/fa3-scheduler-metadata-side-stream.result.json`, `muse_attention_decode/hypotheses/done/regression/fa3-scheduler-metadata-side-stream.patch`; branch `hyp/muse_attention_decode/fa3-scheduler-metadata-side-stream`
