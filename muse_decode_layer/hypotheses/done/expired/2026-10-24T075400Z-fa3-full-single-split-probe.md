---
problem: muse_decode_layer
slug: fa3-full-single-split-probe
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  Sensitivity probe re-minted from the held fa3-num-splits-full-no-metadata review: an explicit
  num_splits can only CAP FA3's device-side dynamic split (min(dynamic, static)), never raise it,
  so this caps BELOW the dynamic pick. On full-attention layers only (cfg['window'] == 0, both
  q_len=1 decode and q_len=8 verify), call flash_attn_with_kvcache with num_splits=1 and
  scheduler_metadata=None (the judge's metadata encodes its own split schedule built with
  num_splits=0, so it must be dropped for the cap to take effect; verify blocks already pass
  None). Every other argument of the call, the sliding branch and everything outside the FA3 call
  stay exactly as on master. Purpose: learn whether full decode attention at page size 1 is
  gather-pattern-bound or parallelism (memory-level-parallelism) bound before spending runs on a
  custom kernel (triton-flash-decode-full-paired-kv) or trtllm-decode-full-attention. Grids become
  64 CTAs at B=32, 32 at B=16, 2 at B=1 (no partials, no FlashAttnFwdCombine). Reading: if
  decode_32 full FA3 stays near its 491.6 us (1.33 TB/s) with half the CTAs, per-CTA gather
  latency/pattern is the limiter and the fix must change the access pattern (both heads' 128-byte
  row per token, deeper pipelining); if it roughly doubles, in-flight parallelism matters and a
  kernel that oversubscribes the SMs (more splits) is the lever. This is a check-run diagnostic
  and is expected to lose on the B=1 full shapes.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: "the diagnostic: 64 CTAs instead of ~128; flat means gather-pattern-bound, ~2x FA3 time (~+490 us) means MLP-bound; either is a large readable move vs a 491.6 us FA3"
  - shapes: "*layer=full load=decode_16*"
    direction: slower
    why: "32 CTAs on 132 SMs: under a quarter wave, slower unless the per-CTA path can pull 4x its current bandwidth"
  - shapes: "*layer=full load=verify_32*"
    direction: flat
    why: same probe with 8 queries per request (metadata already None); compare its move with decode_32 to see whether the q_len=8 tile changes the bound
  - shapes: "*layer=full load=verify_16*"
    direction: slower
    why: 32 CTAs, as decode_16
  - shapes: "*layer=full load=decode_1*"
    direction: slower
    why: "2 CTAs over 625 key tiles of 128: FA3 18 us -> roughly 10x, minus the 6.9 us combine"
  - shapes: "*layer=full load=verify_1*"
    direction: slower
    why: 2 CTAs, as decode_1
  - shapes: "*layer=sliding*"
    direction: flat
    why: sliding branch untouched; control
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-24T092405Z
review_companions: null
hold_reason: "adversarial review: The probe is sound and would give a clear reading, but neither reading can change a decision that is open right now. The rewrite it is meant to inform (triton-f"
outcome: expired
verdict_detail: "hold expired after 36 h: adversarial review: The probe is sound and would give a clear reading, but neither reading can change a decision that is"
expired_at: 2026-10-25T211328Z
---

# fa3-full-single-split-probe

## Change

Sensitivity probe re-minted from the held fa3-num-splits-full-no-metadata review: an explicit num_splits can only CAP FA3's device-side dynamic split (min(dynamic, static)), never raise it, so this caps BELOW the dynamic pick. On full-attention layers only (cfg['window'] == 0, both q_len=1 decode and q_len=8 verify), call flash_attn_with_kvcache with num_splits=1 and scheduler_metadata=None (the judge's metadata encodes its own split schedule built with num_splits=0, so it must be dropped for the cap to take effect; verify blocks already pass None). Every other argument of the call, the sliding branch and everything outside the FA3 call stay exactly as on master. Purpose: learn whether full decode attention at page size 1 is gather-pattern-bound or parallelism (memory-level-parallelism) bound before spending runs on a custom kernel (triton-flash-decode-full-paired-kv) or trtllm-decode-full-attention. Grids become 64 CTAs at B=32, 32 at B=16, 2 at B=1 (no partials, no FlashAttnFwdCombine). Reading: if decode_32 full FA3 stays near its 491.6 us (1.33 TB/s) with half the CTAs, per-CTA gather latency/pattern is the limiter and the fix must change the access pattern (both heads' 128-byte row per token, deeper pipelining); if it roughly doubles, in-flight parallelism matters and a kernel that oversubscribes the SMs (more splits) is the lever. This is a check-run diagnostic and is expected to lose on the B=1 full shapes.

## Expected

- `*layer=full load=decode_32*` flat: the diagnostic: 64 CTAs instead of ~128; flat means gather-pattern-bound, ~2x FA3 time (~+490 us) means MLP-bound; either is a large readable move vs a 491.6 us FA3
- `*layer=full load=decode_16*` slower: 32 CTAs on 132 SMs: under a quarter wave, slower unless the per-CTA path can pull 4x its current bandwidth
- `*layer=full load=verify_32*` flat: same probe with 8 queries per request (metadata already None); compare its move with decode_32 to see whether the q_len=8 tile changes the bound
- `*layer=full load=verify_16*` slower: 32 CTAs, as decode_16
- `*layer=full load=decode_1*` slower: 2 CTAs over 625 key tiles of 128: FA3 18 us -> roughly 10x, minus the 6.9 us combine
- `*layer=full load=verify_1*` slower: 2 CTAs, as decode_1
- `*layer=sliding*` flat: sliding branch untouched; control

## Risk

This run is expected to lower the overall speedup (it is information, not a candidate for master: never promote it). The length draws differ per shape (spread=agentic, per-shape seeds), so compare each full shape with its own base row, not decode with verify. Confirm in prepare() via a print of sgl_kernel.__version__ that the judge's FA3 accepts num_splits with cache_seqlens (varlen dynamic split); if num_splits=1 with scheduler_metadata=None is rejected, the run fails fast in the tests. The 1M-token full verify test (batch 1, ctx 1048576, q_len 8) runs at 2 CTAs: 8192 key tiles serially per CTA, a few ms, well inside the 15-minute budget. Correctness: no split means no fp32 combine; numerics only improve.

## Adversarial review

Reviewed 2026-10-24T092405Z by an adversarial subagent (model opus).

**Verdict: hold.** The probe is sound and would give a clear reading, but neither reading can change a decision that is open right now. The rewrite it is meant to inform (triton-flash-decode-full-paired-kv: paired 128-byte rows plus about 2 waves of CTAs) is already running and covers both levers, and decode has 4 checks left. Re-queue it only if that rewrite's stage 1 fails to beat FA3's own kernel time on full decode.

### Justification
A. The mechanism is now correct. Under varlen dynamic split an explicit num_splits only caps the split count (min(dynamic, static)), so num_splits=1 does bind: the grid becomes B*h_k CTAs (64 at B=32, 32 at B=16, 2 at B=1) with no partials and no combine. Dropping scheduler_metadata on decode is needed for the cap to apply; it adds an in-call prepare launch of about 2 us, which is small next to the expected moves. The readings are interpretable. At B=32 the CTA count halves and at B=16 it drops 4x, which gives two points against the same 128-CTA base. If the time stays flat, the limit is the global memory system or the gather pattern. If it scales with 1/CTAs, the limit is per-CTA latency or memory-level parallelism. One correction to the premise: the 491.6 us / 1.33 TB/s figure is from the older profile 5755a412. The current best profile (2b1f82408ab64116) shows full-layer FA3 at 374.4 us for decode_32, 188.8 us for decode_16 and 374.0 us for verify_32, which is a uniform ~1.75 TB/s at about 128 CTAs. Readings must use the same-run base rows, not the number in the hypothesis. The expected table is consistent with TASK's full-shape weights. Decode_1 and verify_1 full will lose roughly 10x on an 18-19 us kernel (+150-170 us on a ~184-240 us shape), so the headline drops heavily, as the hypothesis already concedes. Weakest link: decision value, not the physics. Both outcomes argue for the design already running. Pattern-bound means fetch both heads' 128-byte row, which the rewrite does. MLP-bound means oversubscribe with more splits, which the rewrite also does (S=9 at B=32, 17 at B=16). Neither outcome settles trtllm-decode-full-attention either. So a check spent here buys a diagnosis whose only consumer is a contingency, a rewrite stage 1 that underperforms, that has not happened yet. Also, 'flat' does not cleanly separate 'gather pattern' from 'HBM/L2 sector inefficiency of 64-byte head rows', since both are global limits. That matters little because the rewrite targets both.

### Confounders on the current kernel
- marlin-block-m-cap-16: touches only the two Marlin MoE GEMMs, which run after attention on the main stream. It shares no launch, traffic or numerical path with the FA3 call. On full verify_16/32 it changed Marlin time, so compare those rows against the current best's same-run base, not older FINDINGS rows. It is not a confounder of the FA3 delta itself.
- sglang-pipeline-parity-router-hilo-m8: its FA3 call is byte-identical to baseline.py, and its router change runs on the side stream after the o projection. It cannot cause or mask an FA3 split-count effect.

### Interference and A/B needs
No ab-revert or ab-tune is needed, because nothing landed touches the attention call. A control (num_splits=2 as a second point) would sharpen the curve, but it doubles the cost of a probe whose value is already conditional. The run disambiguates itself well enough: the sliding rows are an untouched control, and the full decode and verify rows give CTA ratios of 2x (B=32) and 4x (B=16). The blocker is not ambiguity. It is that the result cannot change today's decisions while the running rewrite answers the practical question directly: does paired-row, oversubscribed decode beat FA3's kernel time? A cheaper diagnostic is also already in flight, since the rewrite's own profile gives achieved TB/s at S=9/17 with 128-byte rows, a direct data point on both levers.

### Open questions
- Re-queue trigger: if triton-flash-decode-full-paired-kv stage 1 is correct but its kernel time on full decode_16/32 does not beat FA3's 188.8/374.4 us, run this probe (num_splits=1, optionally plus num_splits=2) to learn whether the stage 2 fix is more in-flight parallelism or a different access pattern.
- Before any re-queue, read the rewrite's profile. Achieved TB/s at S=9 (B=32) and S=17 (B=16) with 128-byte rows may already answer the gather-vs-MLP question at no quota cost.
- Confirm from the judge's sgl_kernel FA3 source or version print that num_splits=1 with scheduler_metadata=None takes the non-split path (no FlashAttnFwdCombine) and is accepted together with cache_seqlens.
- The hypothesis quotes decode_32 full FA3 at 491.6 us (profile 5755a412), but the current best shows 374.4 us. Update the expected 'why' text before any re-queue so the flat/2x thresholds are read against ~374 us.

