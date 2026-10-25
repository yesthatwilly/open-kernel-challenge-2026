# muse_decode_layer · 2026-10-25T040652Z · check · dense-chain-stage2-attn-side-pdl-t1-gate

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T035414Z-dense-chain-stage2-attn-side-pdl-t1-gate.md` (branch `hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate`).

- **Job:** `c6cb53d8ebc2478a` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `6f0d879`

## Goal
Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its run (b31b8453, check 1.2253x) won full decode_1 by -3.7 us (-2.6%, z-3.9; sliding decode_1 -1.4%, flat) but lost at T=16/32 because dl_o_kernel does not scale past T=1 (O at T=32 40.8 us, 0.41 TB/s; sliding decode_16 +9.2%, decode_32 +13.5%, full decode_32 +6.0%). Change only the predicate that sets/uses state["dl2_ok"] routing in kernel_fn: Q (dl_qkv_kernel<NF>) and O (dl_o_kernel<NF>) run only when tokens == 1; every T in 2..32 and above returns to master attention side (nvjet qkv + FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) byte for byte, with the landed stage-1 dense chain (P, R, Marlin) unchanged. Keep both kernels compiled at NF=1 only if the template allows (drop the NF=2/4 instantiations to cut build time; otherwise leave them unused). Self-check: keep the existing Q/O/FA3/graph-replay cases at T=1 for both the full and sliding branches; drop T>1 cases. Print PATH CHOSEN per T so the log shows Q/O only at T=1. First measurement is an okc check.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree. The worktree already carried the parent's (dense-chain-stage2-attn-side-pdl) patch, staged; my edits are the unstaged diff (+70/-15).

- Gate (review fix applied, as the reviewer and the spec instructed): `DL2_MAX_TOKENS = 32 -> 1` and `DL2_SLIDING1 = "raw" -> "nvjet"`. `_dl2_applies()` is unchanged: tokens > 1 returns False, and sliding B=1 L=1 returns False under "nvjet". Q and O therefore run only at full-attention decode_1. Sliding decode_1, decode_16/32, verify_* and every other shape run `_master_attention` byte for byte, with the stage-1 chain (P, R, Marlin) unchanged. Sliding decode_1 is the in-run drift control.
- Deviation (necessary): the review says DL2_MAX_TOKENS also shrinks ws_q/ws_o. It cannot: the kernels stride ws by MAXT = 32 and `dl2_qkv`/`dl2_o` TORCH_CHECK `ws.numel() >= SPLIT * MAXT * ROWS`, so a 1-token ws would fail every launch and turn the whole path off. Added `DL2_WS_TOKENS = 32` (must match MAXT) for the ws sizing; ws is 6*32*2304 and 2*32*HIDDEN floats as before.
- Not done, by instruction ("keep the change that small"): NF=2/4 instantiations remain compiled and unused at runtime (T > 1 never reaches them), no kernel code touched, module name left as the parent's `muse_decode_dense_chain_v2` (the parent already renamed it versus master; the review's rename suggestion is skipped to keep the TU identical to the parent).
- Self-check: cases cut to T=1 only: (B1 L1 sliding kv 1.0), (B1 L1 full kv 1.0), (B1 L1 full kv 0.25); graph-replay chain cases now (1,1,full) and (1,1,sliding); `_dl2_time` at T=1 only.
- Review open question applied (LESSONS 163): new `_dl2_compare()` runs after the self-check; it graph-times (20 launches x 5 replays, best of 3, CUDA events) custom Q+O against master's qkv GEMM + FusedQKNorm + casts + store_cache + o_norm_gate + o GEMM on the same T=1 full inputs (FA3 identical on both paths, left out), prints both, and keeps the custom path only if it wins; otherwise or on any exception, PATH CHOSEN = MASTER everywhere.
- PATH CHOSEN lines printed per shape class: full decode_1 (Q+FA3+O), sliding decode_1 (master nvjet, drift control), T > 1 (master).
- Merge note: this patch conflicts with the route patch (stack-route-topk-pdl-unpool-nf1) in `_DL_CUDA` around line 295 (the dl2 declarations in `_DL_CPP` and the TU); whichever lands second must hand-merge the TU and keep one module name.
- Risks: the prepare-time comparison is noisy at ~20 us totals (a false "master wins" would silently turn the gain off; read the log line); no GPU here, nothing was run.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no host sync in kernel_fn; `_dl2_compare` and the cut self-check are prepare()-only; kernel_fn's gate is still config-only (`_dl2_applies`).
- [x] Nothing derived from `data` is cached across calls: the diff only adds a constant and prepare()-time synthetic data.
- [x] Weights: unchanged, repacked only in prepare() by the parent's code.
- [x] Activations BF16 / KV FP8 / query at most FP8 E4M3: unchanged kernels; no precision edits in the diff.
- [x] Attention sees every key: FA3 untouched, full window; the diff only narrows which shapes use Q/O.
- [x] No timing patches, harness/IO/secrets, module edits, reference import, subprocesses or network: none added (no subprocess/fork/multiprocessing).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] sm_90a-only PTX / wgmma / setmaxnreg: n/a: this build has none (flags `-O3 -arch=sm_90a -lineinfo`, printed by the parent code; the build brief's 9.0a env wrapper applies only to wgmma).
- [x] Bounded mbarrier waits: n/a: no mbarrier changes in this diff.

okc_lint: `OK with warnings` (only host-sync warnings in prepare(), outside kernel_fn's call graph, plus the two uint8-randint dtype warnings in self-check synthetic data, pre-existing). py_compile: rc 0; __pycache__ removed.

## Risks / what could fail
medium
Small single-shape gain (~+0.003-0.005 headline) near check resolution; read the full decode_1 row (err ~0.6 us) and the dl_qkv/dl_o profile rows. The patch also renames the _DL_CUDA module; if stack-route-topk-pdl-unpool-nf1 lands first, rebase onto it and merge the TU by hand.

## Result (after the run)
- job `c6cb53d8ebc2478a` (check), status done, passed True, tests 13/13
- speedup 1.2717x, 95% CI [1.2526, 1.2908], score 17787.62682743311 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/dense-chain-stage2-attn-side-pdl-t1-gate.result.json`, `muse_decode_layer/hypotheses/done/neutral/dense-chain-stage2-attn-side-pdl-t1-gate.patch`; branch `hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate`
