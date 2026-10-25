---
problem: muse_attention_decode
slug: sglang-path-parity
created: 2026-10-23T223444Z
base: master
area: SGLang path parity
change: >
  Replace the starter's plain-PyTorch kernel_fn with SGLang's own pipeline exactly as baseline.py
  runs it, so the submission is the same five-stage sequence SGLang is timed on:
  FusedQKNorm(head_dim, num_q_heads, num_kv_heads, eps=1e-5) from
  sglang.kernels.ops.muse_spark_v12.fused_qk_norm applied to qkv viewed as [B*L, 2304] (with
  positions and the cached [2^21, 64] cos|sin table when rope_theta != 0, else without), then
  k.reshape(tokens, Hkv*64) and v = qkv[:, (Hq+Hkv)*64:(Hq+2*Hkv)*64], both divided by
  k_scale/v_scale only when config kv_scale != 1, cast .to(float8_e4m3fn).view(uint8) and written
  by sglang.kernels.ops.kvcache.kvcache.store_cache into k_cache/v_cache viewed as [-1, Hkv*64]
  uint8 at cache_loc with reserved_skip_index=-1; then
  sgl_kernel.flash_attn.flash_attn_with_kvcache(q.view(tokens, Hq, 64).to(float8_e4m3fn), k_cache,
  v_cache, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
  max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(window-1, 0)
  if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'],
  k_descale/v_descale=scale.reshape(1,1).expand(B, Hkv) only when kv_scale != 1); take out[0] if a
  tuple; then rms_norm_gate(out.reshape(tokens, Hq*64).to(bfloat16), gate=qkv[:, (Hq+2*Hkv)*64:],
  None, head_dim=64, eps=1e-5, has_norm=True) and return .view(B, L, Hq*64). The module-level
  _NORMS and _ROPE caches are keyed on config values (theta, head counts), never on data. Nothing
  else changes. This is the base every other attention hypothesis stacks on; the starter's 0.0023x
  is an fp32 cuBLAS GEMM over every page-table slot and carries no information about any area.
expected:
  - shapes: "*"
    direction: faster
    why: "from 0.0023x to about 1.0x: the starter attends over every slot the page table can hold with an fp32 gemmSN_NN GEMM (85-88% of every shape) plus a gather; SGLang runs FA3 over the FP8 cache"
  - shapes: "*layer=sliding*"
    direction: flat
    why: "relative to SGLang: identical launches (FusedQKNorm, two FP8 casts, store_cache, q cast, FA3, rms_norm_gate), so the paired-ratio median should sit at 1.0x within the CI"
  - shapes: "*layer=full*"
    direction: flat
    why: same FA3 call with the same scheduler_metadata as SGLang; 1.0x within the CI
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-23T224646Z
review_companions: null
hold_reason: "adversarial review: As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, same descale and tuple handling), so churn.py commit-impl (.cla"
outcome: expired
verdict_detail: "hold expired after 46 h: adversarial review: As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, sam"
expired_at: 2026-10-25T211327Z
---

# sglang-path-parity

## Change

Replace the starter's plain-PyTorch kernel_fn with SGLang's own pipeline exactly as baseline.py runs it, so the submission is the same five-stage sequence SGLang is timed on: FusedQKNorm(head_dim, num_q_heads, num_kv_heads, eps=1e-5) from sglang.kernels.ops.muse_spark_v12.fused_qk_norm applied to qkv viewed as [B*L, 2304] (with positions and the cached [2^21, 64] cos|sin table when rope_theta != 0, else without), then k.reshape(tokens, Hkv*64) and v = qkv[:, (Hq+Hkv)*64:(Hq+2*Hkv)*64], both divided by k_scale/v_scale only when config kv_scale != 1, cast .to(float8_e4m3fn).view(uint8) and written by sglang.kernels.ops.kvcache.kvcache.store_cache into k_cache/v_cache viewed as [-1, Hkv*64] uint8 at cache_loc with reserved_skip_index=-1; then sgl_kernel.flash_attn.flash_attn_with_kvcache(q.view(tokens, Hq, 64).to(float8_e4m3fn), k_cache, v_cache, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(window-1, 0) if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'], k_descale/v_descale=scale.reshape(1,1).expand(B, Hkv) only when kv_scale != 1); take out[0] if a tuple; then rms_norm_gate(out.reshape(tokens, Hq*64).to(bfloat16), gate=qkv[:, (Hq+2*Hkv)*64:], None, head_dim=64, eps=1e-5, has_norm=True) and return .view(B, L, Hq*64). The module-level _NORMS and _ROPE caches are keyed on config values (theta, head counts), never on data. Nothing else changes. This is the base every other attention hypothesis stacks on; the starter's 0.0023x is an fp32 cuBLAS GEMM over every page-table slot and carries no information about any area.

## Expected

- `*` faster: from 0.0023x to about 1.0x: the starter attends over every slot the page table can hold with an fp32 gemmSN_NN GEMM (85-88% of every shape) plus a gather; SGLang runs FA3 over the FP8 cache
- `*layer=sliding*` flat: relative to SGLang: identical launches (FusedQKNorm, two FP8 casts, store_cache, q cast, FA3, rms_norm_gate), so the paired-ratio median should sit at 1.0x within the CI
- `*layer=full*` flat: same FA3 call with the same scheduler_metadata as SGLang; 1.0x within the CI

## Risk

Low risk: it is SGLang's code path, imported from the same image baseline.py imports from. Only risk: the _ROPE table (2^21 x 64 x fp32 = 512 MiB) is built lazily on the first sliding call; if the judge captures the graph on the very first call that allocation happens under capture. Build it at import time for theta=500000 (config is fixed per problem) or on the warm-up call to remove the doubt. Expected gate to pass: all 16 tests, since the baseline passes them by construction.

## Adversarial review

Reviewed 2026-10-23T224646Z by an adversarial Fable subagent (model fable).

**Verdict: hold.** As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, same descale and tuple handling), so churn.py commit-impl (.claude/skills/churnsubs/bin/churn.py:1326-1329) will reject it as identical-to-baseline before any run, file it under done/rejected, and _eligible_queue (churn.py:1079-1083) will then permanently strand all nine dependents. A hold keeps it recoverable; the base should be created by a commit, not a run, and the parity measurement it promises already exists in two forms: the judge's own SGLang column on every shape, and the verbatim-baseline check on muse_bf16_proj (job 1ccc6ba13da94194: 1.0028x [0.988, 1.018]).

### Justification
The mechanism is trivially true: the starter's time is one fp32 gemmSN_NN over every page-table slot (2488 us, 85-88% of every shape, FINDINGS.md) plus a 99 us gather, and SGLang's path replaces it with FA3 over the FP8 cache. The already-run fused-preattention-triton check (muse_attention_decode/profiles/submission.dfee6f2939974bad.txt, 1.127x [1.111, 1.144], 16/16) shows what that path costs on this judge: sliding decode-1 is FA3 8.2 us + FlashAttnFwdCombine 2.5 + prepare_varlen 1.8 + pre-attention 1.5 + norm_gate 1.0 = 15.1 us against SGLang's 20.9 us, i.e. the sliding shapes (weight 46, floors 0.1-3.7 us) are launch-latency bound and the full shapes at batch 16/32 (weight 16, 200-390 us vs 63-127 us floors) are the only bandwidth-bound ones, sitting ~3x off the floor. The expected directions are consistent with TASK.md: '*' faster is a property of the starter, and 'flat within the CI' on both layer types is a tautology because the submission is the same code the judge times as SGLang. Weakest link: the hypothesis has no mechanism of its own, so its result cannot be attributed to any area; it is a base commit dressed as a hypothesis, and the pipeline's normalized duplicate check (NOTES.md 'Normalized duplicate check') was written after the first review flagged exactly this file. Secondary weakness: 'all 16 tests pass by construction' is inferred, not observed; no baseline.py job exists locally for this problem (only a8d647772fa945e5 and dfee6f2939974bad), though the judge's SGLang copy must pass the same tests to be timed.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
Nothing is landed on master (landed: []), so ab-revert and ab-tune have no target, and the change has no sub-change to isolate: it is one wholesale replacement whose answer is known within +/-1.5% from the bf16 verbatim-baseline check and from the fact that the judge's 'sglang' column on every shape is this same code timed in the same graph-replay, cold-L2, order-swapped pairing. The only meaningful control relationship runs the other way: this pipeline is the control arm of fused-preattention-triton, and that arm is supplied per shape by the judge for free. A companion would spend a check run to re-measure harness symmetry a third time. No companions.

### Open questions
- Who creates the base? commit-impl exempts only kind: control from the baseline dedup, and a control needs a done parent; filing this as control with controls_for: fused-preattention-triton is circular while fused-preattention-triton depends_on sglang-path-parity. The clean path is a human commit of baseline-equivalent code to master (keep the '# okc: problem=muse_attention_decode' first line) outside churnsubs, then re-point the nine dependents to depends_on: null / base: master. fused-preattention-triton already replaces the whole kernel_fn (working tree, 124 insertions) so it does not need the parity commit to exist at all.
- Why is job dfee6f2939974bad (1.127x [1.111, 1.144], 16/16, profile at muse_attention_decode/profiles/submission.dfee6f2939974bad.txt) absent from BEST.json and FINDINGS.md? It is the real current best on this problem, ran on the uncommitted working-tree submission.py (git status: M muse_attention_decode/submission.py), and the fused-preattention-triton hypothesis says to take that code verbatim; the judge returns the earlier result for identical code, so harvesting it costs no run. Until it is harvested, FINDINGS tells every hypothesis author the time goes to a cuBLAS GEMM, which is false for the actual best.
- Does the judge's own SGLang copy equal the distributed baseline.py for this problem? TASK.md says the judge uses its own copies of reference.py/checks.py; a verbatim baseline.py check on muse_bf16_proj measured 1.0028x, which suggests yes, but it has not been observed for muse_attention_decode.
- After the base lands, BEST.json seeding: best-init attributes a job to the newest master commit at or before its submit time (NOTES.md open questions); dfee6f2939974bad ran against uncommitted code, so its attribution will be approximate unless the harvested hypothesis records impl_commit explicitly.

