# muse_attention_decode · 2026-10-24T104354Z · check · paired-heads-launch-cleanup-ab-tune-batch-gate

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T104027Z-paired-heads-launch-cleanup-ab-tune-batch-gate.md` (branch `hyp/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate`).

- **Job:** `bd64bf9f9e904f02` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `f6513e3`

## Goal
From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'], and use this single predicate both for the pre-attention kernel's q layout (pass it as a constexpr, e.g. PAIRED, plus the q buffer shape [tokens, Hq, 128] vs [tokens, Hq, 64]) and for the FA3/epilogue branch. When paired is false on a full layer (B=1 decode_1/verify_1, and odd Hkv), run exactly master's pre-pairing path. The pre-attention kernel writes the unpadded [tokens, Hq, 64] FP8 q. FA3 gets the original k_cache/v_cache views, descale expanded to (B, Hkv), and scheduler_metadata=data['scheduler_metadata'] (the judge's). The output goes through SGLang's rms_norm_gate on out.reshape(tokens, Hq*64), as before fa3-hdim128-head-pairing. Sliding layers are unchanged. B>=16 full layers are byte-identical to the parent.

## What changed, and where
Implemented inline by churnsubs on the parent's commit (12230f1): the single predicate in kernel_fn became `paired = window == 0 and hkv % 2 == 0 and batch > 1` (batch = cfg["batch"]). The parent already uses that one predicate for the q layout (PAIRED constexpr and buffer shape) and for the FA3/epilogue branch, and its unpaired branch is the pre-pairing path (unpadded FP8 q, original cache views, descale (B, Hkv), the judge's scheduler_metadata, SGLang rms_norm_gate). B>=16 full layers are byte-identical to the parent. No deviation.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate diff` (one predicate line plus a comment).

- [x] CUDA-graph capturable: the edit is on config values only (cfg window, num_kv_heads and batch); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 and queries FP8 E4M3: unchanged by this edit (both paths already exist on the parent and passed review).
- [x] Attention sees every key: unchanged: both branches attend every key (the paired branch reads the same KV rows two heads at a time).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line marker kept; no triton.autotune added.

okc_lint output:

```
CLEAN: .worktrees/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate/muse_attention_decode/submission.py
```

## Risks / what could fail
low
Decides whether pairing must be gated to batch > 1 or whether the launch cleanup alone makes batch 1 acceptable; also the likely promotion candidate (~+4-5% geomean over the ungated cleanup if the estimates hold).

## Result (after the run)
- job `bd64bf9f9e904f02` (check), status done, passed True, tests 16/16
- speedup 1.2686x, 95% CI [1.2496, 1.2877], score 2968.7522922676703 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.1705932092473075, "ci": [1.1530343111085979, 1.1923944446416679], "job": "facc479992e24a35"}; base result: muse_attention_decode/hypotheses/done/gain/paired-heads-launch-cleanup.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.result.json`, `muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.patch`; branch `hyp/muse_attention_decode/paired-heads-launch-cleanup-ab-tune-batch-gate`
