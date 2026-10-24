---
problem: muse_attention_decode
slug: fa3-scheduler-metadata-full-verify
created: 2026-10-23T223722Z
base: master
area: FA3 call
change: >
  On full-attention verify blocks (config window == 0 and q_len == 8), where the judge passes
  scheduler_metadata=None (SGLang leaves the schedule to each call), build FA3's tile schedule
  inside kernel_fn and pass it: call sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B,
  max_seqlen_q=8, max_seqlen_k=page_table.shape[1]*page_size (a shape, not a value; the pool is
  sized for the longest possible length), num_heads=Hq, num_heads_k=Hkv, headdim=64,
  cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
  page_size=page_size, causal=True, num_splits=0) right before flash_attn_with_kvcache and pass
  its result as scheduler_metadata. It is a device-side kernel over cache_seqlens (no host sync;
  SGLang captures it in its own CUDA graphs), so it adds one small launch. Sliding layers and
  decode steps (which already carry the judge-built metadata) are untouched. The mechanism: with
  scheduler_metadata FA3 uses its precomputed dynamic split schedule balanced over the batch's
  actual lengths; without it, the verify path relies on the static heuristic, and at batch 16/32
  with 80K contexts (spread agentic, lengths 74K-86K) a balanced schedule may finish the longest
  tiles sooner.
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us floor; the decode_16 shape, which has the metadata, runs 201 us for the same bytes at q_len 1, so the metadata is worth testing at q_len 8
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: 390 us vs 127 us floor, same mechanism
  - shapes: "*layer=full load=verify_1*"
    direction: slower
    why: "one extra launch (~2-3 us) on a 35.6 us shape with 2 tiles: little to balance"
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched
risk: medium
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T120131Z
review_companions: null
hold_reason: "adversarial review: Master has made this moot. Full verify at batch 16/32 now runs on the head-paired path with scheduler_metadata=None, and so does full decode at batch > 1. The p"
---

# fa3-scheduler-metadata-full-verify

## Change

On full-attention verify blocks (config window == 0 and q_len == 8), where the judge passes scheduler_metadata=None (SGLang leaves the schedule to each call), build FA3's tile schedule inside kernel_fn and pass it: call sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=8, max_seqlen_k=page_table.shape[1]*page_size (a shape, not a value; the pool is sized for the longest possible length), num_heads=Hq, num_heads_k=Hkv, headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) right before flash_attn_with_kvcache and pass its result as scheduler_metadata. It is a device-side kernel over cache_seqlens (no host sync; SGLang captures it in its own CUDA graphs), so it adds one small launch. Sliding layers and decode steps (which already carry the judge-built metadata) are untouched. The mechanism: with scheduler_metadata FA3 uses its precomputed dynamic split schedule balanced over the batch's actual lengths; without it, the verify path relies on the static heuristic, and at batch 16/32 with 80K contexts (spread agentic, lengths 74K-86K) a balanced schedule may finish the longest tiles sooner.

## Expected

- `*layer=full load=verify_16*` faster: 205 us vs 63 us floor; the decode_16 shape, which has the metadata, runs 201 us for the same bytes at q_len 1, so the metadata is worth testing at q_len 8
- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor, same mechanism
- `*layer=full load=verify_1*` slower: one extra launch (~2-3 us) on a 35.6 us shape with 2 tiles: little to balance
- `*layer=sliding*` flat: untouched
- `*load=decode_*` flat: untouched

## Risk

Medium risk: max_seqlen_k must be a Python int derived from shapes, never cache_seqlens.max() (a host sync, which breaks graph capture). The verify_1 full shape will likely lose one launch's worth; the load-level outcome is decided by whether verify_16/32 full shapes (2.4-3.4x the sliding shapes' load time) gain more than that. If get_scheduler_metadata's result makes no difference at q_len 8 (FA3 may ignore it when max_seqlen_q > 1 or when it is not a decode), the outcome is 'slower by one launch everywhere it is called' and the hypothesis is rejected cleanly.

## Adversarial review

Reviewed 2026-10-24T120131Z by an adversarial subagent (model opus).

**Verdict: hold.** Master has made this moot. Full verify at batch 16/32 now runs on the head-paired path with scheduler_metadata=None, and so does full decode at batch > 1. The profile shows FA3 already runs prepare_varlen_num_blocks_kernel inside each of those calls, so a precomputed schedule only moves the same on-device scheduling kernel into a separate launch. The expected gain has no mechanism behind it.

### Justification
A: The premise is that without scheduler_metadata FA3 falls back to a static heuristic, and that a precomputed schedule balanced over the actual lengths would finish the longest tiles sooner. The current-best profile in FINDINGS contradicts this. Every full verify shape (verify_1 2.8 us, verify_16 3.2 us, verify_32 in the top-kernel lists) already runs flash::prepare_varlen_num_blocks_kernel<..., true>. That is the dynamic-split scheduler get_scheduler_metadata would launch, now running inside the FA3 call. LESSONS records the same thing: FA3 already sizes splits on the GPU under varlen. With metadata passed, FA3 skips its internal prepare kernel. It then uses the schedule from the same function, given the same inputs: cache_seqlens, and max_seqlen_k = page_table.shape[1]*page_size, which is also what FA3 derives internally for paged KV when none is given. So the schedule would match and the result would be flat or slightly slower: a separate launch, the Python wrapper and a metadata allocation, against a removed in-call launch of the same cost. The evidence offered was 'decode_16, which has the metadata, runs 201 us for the same bytes at q_len 1'. That is SGLang's number (201 vs 205 us), so it never isolated metadata. On master, full decode_16 also runs without metadata (paired, metadata=None) at 155.9 us against verify_16's 163.5 us. The gap is about 5%, explained by 8x the queries and causal masking, and the reference point the hypothesis relies on no longer exists. The geometry is also stale. The change says headdim=64 and num_heads_k=Hkv, but master's batch > 1 full path calls FA3 on a paired view (hdim 128, S = Hkv/2 KV heads, q padded to 128). A schedule built for the unpaired geometry would at best be ignored. At worst it mis-sizes the splits or tile counts for the actual call, which is a correctness risk. Master's own comment drops the judge's metadata for exactly this reason. verify_1 stays unpaired on master, but its expected direction is 'slower', so it cannot carry the result. Weakest link: the claim that the no-metadata path uses a static heuristic. The profile shows the dynamic scheduler running in-call.

### Confounders on the current kernel
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup-ab-tune-batch-gate: full verify_16/32 now goes through the paired view (hdim 128, Hkv/2 heads) with scheduler_metadata deliberately None. The spec's get_scheduler_metadata arguments (headdim=64, num_heads_k=Hkv) describe a geometry that no longer exists at batch > 1. Any result would mostly reflect a mismatch between the schedule and the call, not the effect of a balanced schedule.
- paired-heads-launch-cleanup: it also changed what the decode_16 comparison point is. Full decode at batch > 1 now runs without metadata too, so 'decode has the metadata' is no longer true on master.
- fused-preattention-triton: it touches neither the FA3 scheduling path nor the metadata, so it does not confound this change.

### Interference and A/B needs
C: No companion is warranted, because the parent result itself is moot. Correctly rebuilt for the paired geometry (headdim=128, num_heads_k=Hkv//2, num_heads=Hq), it would reproduce the schedule FA3 already computes in-call, so expect flat within the ~1.5-2.5% drift. An ab-revert of the pairing would test the change on a lineage that master no longer runs at these shapes, and could not change a decision. A control would have no sub-change to isolate: the change has only one, and the in-call prepare kernel already acts as that control. Spending a check here buys nothing, while the stronger levers for full verify (fa3-batched-q-verify, triton-flash-decode-full-verify) are queued.

### Open questions
- If anyone revives this, first confirm in sgl_kernel's flash_api that mha_fwd, given scheduler_metadata=None with varlen paged KV, calls prepare_varlen_num_blocks with the same num_splits heuristic and max_seqlen_k as get_scheduler_metadata would. The profile strongly suggests it does. Any remaining difference would be in pack_gqa or num_splits choices, which can be tested more cheaply as a num_splits/pack_gqa argument tweak, and the fa3-num-splits-* items were held for the same reason.
- If it is kept for verify_1 only (unpaired), the hypothesis's own expectation there is 'slower', so it has no upside case.

