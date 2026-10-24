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
status: queued
review_at: 2026-10-24T120904Z
review_companions: null
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

