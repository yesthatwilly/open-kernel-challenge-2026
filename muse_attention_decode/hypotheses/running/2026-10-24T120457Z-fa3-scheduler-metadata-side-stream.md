---
problem: muse_attention_decode
slug: fa3-scheduler-metadata-side-stream
created: 2026-10-24T120457Z
base: master
area: FA3 call (schedule, off critical path)
change: >
  Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape,
  full paired shapes at batch > 1, full verify_1), FA3 runs
  flash::prepare_varlen_num_blocks_kernel inside the call, serially after _pre_attention_kernel
  (1.7-1.8 us on sliding, 2.8-3.6 us on full; see the batch-gate profile bd64bf9f9e904f02). That
  kernel reads only cache_seqlens / cu_seqlens_q and shapes, not anything the pre-attention kernel
  writes, so build the same schedule concurrently. In kernel_fn, before the _pre_attention_kernel
  launch: cur = torch.cuda.current_stream(); _SIDE (one module-level torch.cuda.Stream() created
  lazily per device, never derived from data).wait_stream(cur); under torch.cuda.stream(_SIDE):
  meta = sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=L,
  max_seqlen_k=page_table.shape[1]*page_size, num_heads=Hq, num_heads_k=attn_heads (Hkv unpaired,
  Hkv/2 paired), headdim=2*d if paired else d, headdim_v same, cache_seqlens=cache_seqlens,
  qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True,
  window_size=(window-1, 0) if window else (-1, -1), num_splits=0, pack_gqa=None, sm_margin=0),
  i.e. exactly the arguments flash_attn_with_kvcache passes to its internal scheduler for this
  call (check the installed sgl_kernel flash_attn.py signature and mirror its defaults; if any
  argument such as max_seqlen_k or has_softcap differs, use what the in-call path derives). Then
  launch _pre_attention_kernel on cur as now, cur.wait_stream(_SIDE), and pass
  scheduler_metadata=meta to FA3. Also record meta on cur (meta.record_stream(cur)) so the
  allocator does not reuse it. Full decode_1 keeps the judge-built metadata unchanged (it already
  has no prepare kernel in its profile). Nothing else changes: query layout, pairing gate,
  epilogues, descale. This differs from the held fa3-scheduler-metadata-full-verify: that one
  launched the same scheduler serially (moving cost, not removing it) and used the unpaired
  geometry; this one runs it on a forked graph branch in parallel with the pre-attention kernel
  and uses the geometry of the actual call.
expected:
  - shapes: "*layer=sliding*"
    direction: faster
    why: "the 1.7-1.8 us prepare kernel overlaps the 1.7-4.2 us pre-attention kernel instead of following it: ~1.5 us of 16.6-28.9 us, 5-9%"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 3.1 us prepare off the critical path, ~2%
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: 3.5 us prepare off the critical path, ~1%
  - shapes: "*layer=full load=verify_*"
    direction: faster
    why: 2.8-3.6 us prepare off the critical path, 1-9% (verify_1 most)
  - shapes: "*layer=full load=decode_1 *"
    direction: same
    why: keeps the judge metadata; untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T120904Z
review_companions: null
hold_reason: null
stage: implemented
claimed_at: 2026-10-24T124117Z
base_sha: 10d039a6478096a1360009f69d62cd7b1c17f800
base_blob: efc544103e6d303d1f8602caa9439628043a17a7
branch: hyp/muse_attention_decode/fa3-scheduler-metadata-side-stream
worktree: .worktrees/muse_attention_decode/fa3-scheduler-metadata-side-stream
mode: check
implemented_at: 2026-10-24T124759Z
impl_commit: 83e8e876b2b559d7cd2f185d528af527d4c32dd0
submission_sha256: 965e87bd0e4400924d89aa7f49bb3079e4df0e0302effadd93d2d662a47cab4a
norm_sha256: c57e179c589b5624b63de22d734c7222680687561b3371f17fbba433844bcf28
submission_blob: 35854e68c3519cf7cd8e08030ed8c5f1af9983eb
lint_warnings: 0
attempt_file: 2026-10-24T124759Z-check-fa3-scheduler-metadata-side-stream.md
prepared_at: 2026-10-24T124759Z
---

# fa3-scheduler-metadata-side-stream

## Change

Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash::prepare_varlen_num_blocks_kernel inside the call, serially after _pre_attention_kernel (1.7-1.8 us on sliding, 2.8-3.6 us on full; see the batch-gate profile bd64bf9f9e904f02). That kernel reads only cache_seqlens / cu_seqlens_q and shapes, not anything the pre-attention kernel writes, so build the same schedule concurrently. In kernel_fn, before the _pre_attention_kernel launch: cur = torch.cuda.current_stream(); _SIDE (one module-level torch.cuda.Stream() created lazily per device, never derived from data).wait_stream(cur); under torch.cuda.stream(_SIDE): meta = sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=L, max_seqlen_k=page_table.shape[1]*page_size, num_heads=Hq, num_heads_k=attn_heads (Hkv unpaired, Hkv/2 paired), headdim=2*d if paired else d, headdim_v same, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, window_size=(window-1, 0) if window else (-1, -1), num_splits=0, pack_gqa=None, sm_margin=0), i.e. exactly the arguments flash_attn_with_kvcache passes to its internal scheduler for this call (check the installed sgl_kernel flash_attn.py signature and mirror its defaults; if any argument such as max_seqlen_k or has_softcap differs, use what the in-call path derives). Then launch _pre_attention_kernel on cur as now, cur.wait_stream(_SIDE), and pass scheduler_metadata=meta to FA3. Also record meta on cur (meta.record_stream(cur)) so the allocator does not reuse it. Full decode_1 keeps the judge-built metadata unchanged (it already has no prepare kernel in its profile). Nothing else changes: query layout, pairing gate, epilogues, descale. This differs from the held fa3-scheduler-metadata-full-verify: that one launched the same scheduler serially (moving cost, not removing it) and used the unpaired geometry; this one runs it on a forked graph branch in parallel with the pre-attention kernel and uses the geometry of the actual call.

## Expected

- `*layer=sliding*` faster: the 1.7-1.8 us prepare kernel overlaps the 1.7-4.2 us pre-attention kernel instead of following it: ~1.5 us of 16.6-28.9 us, 5-9%
- `*layer=full load=decode_16*` faster: 3.1 us prepare off the critical path, ~2%
- `*layer=full load=decode_32*` faster: 3.5 us prepare off the critical path, ~1%
- `*layer=full load=verify_*` faster: 2.8-3.6 us prepare off the critical path, 1-9% (verify_1 most)
- `*layer=full load=decode_1 *` same: keeps the judge metadata; untouched

## Risk

Correctness risk: a schedule built for a different geometry than the call (headdim, num_heads_k, window, page_size, max_seqlen_k) can mis-size splits; the reviewer should confirm against sgl_kernel flash_api mha_fwd how it derives max_seqlen_k and pack_gqa for paged KV and that a passed metadata tensor fully replaces the in-call prepare (full decode_1 profile shows no prepare kernel when metadata is passed). Perf risk: a graph fork/join adds edges, not events, but if FA3 PDL overlap of prepare with pre-attention already hides part of it (GPU busy rows are ~1 us under the kernel sums) the gain shrinks to under 1 us. Not input-derived caching: the schedule is rebuilt every call; only the stream object is module-level. If the running fa3-window-trimmed-page-table lands, the sliding metadata must use its trimmed page table width and lengths.

## Adversarial review

Reviewed 2026-10-24T120904Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is sound and legal: a passed scheduler_metadata removes FA3's in-call prepare_varlen_num_blocks_kernel (full decode_1 with the judge's metadata shows no prepare row), and a forked capture branch can build the same schedule while _pre_attention_kernel runs. The realistic saving is smaller than claimed, about 0.5-1.5 us per sliding shape, because the 1.7-1.8 us prepare row is partly hidden already. Sliding shapes carry a readable signal; full batch>1 shapes are inside drift. One run is interpretable from the per-kernel profile, so no companion is needed. Dispatch after fa3-window-trimmed-page-table harvests.

### Justification
Mechanism: in master's bd64bf9f9e904f02 profile, every call made with scheduler_metadata=None has a flash::prepare_varlen_num_blocks_kernel row: all six sliding shapes (1.7-1.8 us), full verify_1 (2.8 us, unpaired) and full batch-16/32 decode/verify (2.9-3.6 us, paired). Full decode_1 receives the judge's metadata and has no prepare row (24.0 us kernel sum), which confirms that a passed metadata tensor fully replaces the in-call prepare (FA3 sets skip_scheduler_metadata_computation). The prepare kernel reads only cache_seqlens, cu_seqlens_q and host-side shape scalars; _pre_attention_kernel writes q and the new cache rows and never touches those. So there is a true data independence, and fork/join on a side stream is a legal CUDA-graph branch. The same pattern already passed this judge in muse_decode_layer master (router side stream: wait_stream fork, torch.cuda.stream, wait_stream join). Size: on sliding shapes the GPU-busy row is 1.1-1.5 us below the kernel sum, and the tool notes that an overlapping kernel's row includes its wait. So part of the prepare row is already overlapped through FA3's PDL launch of the main kernel after prepare. Taking prepare off the in-stream path also gives up that prepare->main PDL pairing, and adds one fork and one join edge plus a second-branch node (LESSONS: about 0.3 us per extra Triton-sized node). The realistic saving is 0.5-1.5 us on sliding: 3-9% at decode_1/verify_1, which carry 16.6-17.2 us, and 2-5% at batch 16/32. On full layers it is 1-3 us on 156-302 us, i.e. 0.3-2%, below the 1.5-2.5% per-shape drift. Full verify_1 (30.2 us, 2.8 us prepare) is the one full shape that can read clearly. Load arithmetic with the TASK weights (46 x sliding + 16 x full): about 1 us sliding and 2 us full give roughly +3.8% decode_1, +6% verify_1, +2.2% decode_16/verify_16 and +1.3% decode_32/verify_32, so a geomean around +2.5-3%. That is just above the ±1.5% CI. The expected directions are consistent with this. The sliding magnitudes (5-9%) are the optimistic end. Full decode_1 'same' is correct. Weakest link: equating the profiled 1.7-3.6 us prepare row with its critical-path cost. Part of it is already hidden by PDL, and the fork/join is not free. If FA3's PDL already hides most of it, the net is under 0.5 us and flat. A second weak point: the gain assumes the explicitly built schedule is byte-identical to the in-call one. Any argument mismatch (headdim 128 vs 64, num_heads_k S vs Hkv, window adjustment, page_size, max_seqlen_k, pack_gqa, has_softcap, sm_margin) changes num_splits or tile counts. On the 150-290 us full main kernel that would swamp a 3 us saving in either direction, and it can be a correctness bug, not only a perf one.

### Confounders on the current kernel
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup-ab-tune-batch-gate: these create the opportunity on full layers. Pairing at batch>1 dropped the judge's unpaired metadata, so FA3 pays an in-call prepare that SGLang and master-unpaired avoid. This is a dependency, not a confounder of attribution: with pairing reverted, full batch>1 would use the judge metadata and the change would not apply there. The real risk is geometry. The schedule must be built for the paired call (headdim 2*d, num_heads_k Hkv/2, padded q) at batch>1 and for the unpaired call at verify_1. Derive every argument from the actual q_attn/k_attn/page_table passed to FA3, never from cfg alone, so a future gate change cannot silently desynchronise them.
- fused-preattention-triton: it sets the overlap window. Prepare can only hide under the single 1.4-4.2 us _pre_attention_kernel. The saving is capped by min(prepare, pre-attention) minus fork/join cost, so at batch 1, where pre-attention is about 1.5-2.1 us, the hidden fraction is smallest. It does not touch the scheduling path, so it is not a confounder of attribution.
- paired-heads-launch-cleanup: it does not touch the FA3 scheduling path. The paired epilogue runs after FA3 and is unaffected. No confound.
- In flight, not landed: fa3-window-trimmed-page-table is in the judge now. It changes the sliding page_table width (hence max_seqlen_k) and possibly the sliding lengths FA3 sees. If it lands, this change must be built on it and the sliding metadata must use the trimmed tensors, otherwise the schedule mismatches the call. Measured on pre-trim master, the sliding gain would not transfer automatically. Wait for its harvest before claiming.

### Interference and A/B needs
No companion. The obvious control is the same metadata built serially on the current stream just before FA3, with no side stream, which is the re-geometried fa3-scheduler-metadata-full-verify. It would separate 'overlap' from 'a different schedule'. But the single run's own profile already answers that, more cheaply. (1) If the schedule is identical, the FA3 main-kernel and FlashAttnFwdCombine rows stay within about 1% of bd64bf9f9e904f02 on every shape (Triton and FA3 rows reproduce within about 1%), and the prepare row still appears, with the same template arguments (<1,false>/<2,false> sliding, <1,true>/<2,true> full), now on a forked branch. (2) If the main or combine rows move by more than about 2% on sliding, or more than about 1% on full, the schedule differs, and the result is attributable to that and not to the overlap. (3) A sliding-shape headline change with unchanged main/combine rows is attributable to the overlap alone. A loss with unchanged rows means fork/join plus lost PDL costs more than the hidden prepare. That is also unambiguous, and it settles the idea. Neither a gain nor a loss is ambiguous given the per-kernel rows, so a second check (attention has 5 left today, one of them committed to fa3-window-trimmed-page-table) is not warranted. Read sliding shapes and full verify_1 for the verdict. Treat full batch-16/32 rows as an in-run sanity check only: their expected 1-2% sits within drift.

### Open questions
- Does the installed sgl_kernel get_scheduler_metadata accept and forward window_size (and attention_chunk/has_softcap), and does it apply mha_fwd's causal/local normalisation identically? If it lacks window_size, the sliding half cannot be built correctly and should be dropped, leaving the full-layer half (and verify_1).
- Is the FA3 main kernel launched with PDL after prepare in this build? If so, how much of the 1.7 us sliding prepare row is already hidden? The GPU-busy vs kernel-sum gap of 1.1-1.5 us on sliding suggests much of it, which would make the net saving under 1 us.
- Sequencing: fa3-window-trimmed-page-table is in the judge now. If it lands, rebuild this on it and size the sliding metadata from the trimmed page_table. If it is not landed, build on current master. Do not claim before it harvests.
- The expected-shape globs use 'load=decode_1' while profile case keys print 'load=decode 1'. Confirm the harvest's glob matcher normalises spaces to underscores, or the per-shape expectations will not match.


## Implementation notes

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

## Rules check

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
