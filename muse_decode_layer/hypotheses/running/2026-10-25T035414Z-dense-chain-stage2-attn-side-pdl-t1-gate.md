---
problem: muse_decode_layer
slug: dense-chain-stage2-attn-side-pdl-t1-gate
created: 2026-10-25T035414Z
base: master
area: qkv GEMM + o GEMM + attention-side small kernels (T=1 only)
change: >
  Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and
  narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its run (b31b8453, check 1.2253x)
  won full decode_1 by -3.7 us (-2.6%, z-3.9; sliding decode_1 -1.4%, flat) but lost at T=16/32
  because dl_o_kernel does not scale past T=1 (O at T=32 40.8 us, 0.41 TB/s; sliding decode_16
  +9.2%, decode_32 +13.5%, full decode_32 +6.0%). Change only the predicate that sets/uses
  state["dl2_ok"] routing in kernel_fn: Q (dl_qkv_kernel<NF>) and O (dl_o_kernel<NF>) run only
  when tokens == 1; every T in 2..32 and above returns to master attention side (nvjet qkv +
  FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) byte for byte, with the
  landed stage-1 dense chain (P, R, Marlin) unchanged. Keep both kernels compiled at NF=1 only if
  the template allows (drop the NF=2/4 instantiations to cut build time; otherwise leave them
  unused). Self-check: keep the existing Q/O/FA3/graph-replay cases at T=1 for both the full and
  sliding branches; drop T>1 cases. Print PATH CHOSEN per T so the log shows Q/O only at T=1.
  First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: stage 2 measured -3.7 us (-2.6%, z-3.9) at full decode_1 (Q 8.4 us at 4.5 TB/s, O 6.4 us at T=1)
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: stage 2 read -1.4% here, within err
  - shapes: "*load=decode_16 * *load=decode_32 * *load=verify_*"
    direction: flat
    why: back on master attention side; removes the +6-13.5% dl_o_kernel losses
risk: medium
depends_on: dense-chain-stage2-attn-side-pdl
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T040253Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T040300Z
base_sha: 6fbe7b720d0d1b0ee3aa210d94a1e35d73a2a554
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate
worktree: .worktrees/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate
mode: check
restack_of: dense-chain-stage2-attn-side-pdl
stale_base: null
implemented_at: 2026-10-25T040651Z
impl_commit: 6f0d879cb114d99264c5ba7ace8e44fb19e852d3
submission_sha256: ef96d03b7129c85c6475bebf7126ef85ae74ad2c5966cb5dfa029446f195e90c
norm_sha256: 605ff6b120af7d930a0675df6bece48633757a4ca84a016364f3e8bfd69ead0d
submission_blob: c5a4cb4270f0f3704ebd50d20e3598670c6e8deb
lint_warnings: 34
attempt_file: 2026-10-25T040652Z-check-dense-chain-stage2-attn-side-pdl-t1-gate.md
prepared_at: 2026-10-25T040652Z
---

# dense-chain-stage2-attn-side-pdl-t1-gate

## Change

Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its run (b31b8453, check 1.2253x) won full decode_1 by -3.7 us (-2.6%, z-3.9; sliding decode_1 -1.4%, flat) but lost at T=16/32 because dl_o_kernel does not scale past T=1 (O at T=32 40.8 us, 0.41 TB/s; sliding decode_16 +9.2%, decode_32 +13.5%, full decode_32 +6.0%). Change only the predicate that sets/uses state["dl2_ok"] routing in kernel_fn: Q (dl_qkv_kernel<NF>) and O (dl_o_kernel<NF>) run only when tokens == 1; every T in 2..32 and above returns to master attention side (nvjet qkv + FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) byte for byte, with the landed stage-1 dense chain (P, R, Marlin) unchanged. Keep both kernels compiled at NF=1 only if the template allows (drop the NF=2/4 instantiations to cut build time; otherwise leave them unused). Self-check: keep the existing Q/O/FA3/graph-replay cases at T=1 for both the full and sliding branches; drop T>1 cases. Print PATH CHOSEN per T so the log shows Q/O only at T=1. First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: stage 2 measured -3.7 us (-2.6%, z-3.9) at full decode_1 (Q 8.4 us at 4.5 TB/s, O 6.4 us at T=1)
- `*layer=sliding load=decode_1 *` flat: stage 2 read -1.4% here, within err
- `*load=decode_16 * *load=decode_32 * *load=verify_*` flat: back on master attention side; removes the +6-13.5% dl_o_kernel losses

## Risk

Small single-shape gain (~+0.003-0.005 headline) near check resolution; read the full decode_1 row (err ~0.6 us) and the dl_qkv/dl_o profile rows. The patch also renames the _DL_CUDA module; if stack-route-topk-pdl-unpool-nf1 lands first, rebase onto it and merge the TU by hand.

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** The diagnosis is right: dl_o_kernel does not scale past T=1 (0.41 TB/s at T=32), and narrowing to T=1 keeps the one measured win (full decode_1 -3.7 us, z -3.9) while removing the +6 to +13.5% losses. Sharpen the gate to full-attention T=1 only (DL2_SLIDING1='nvjet'), so sliding decode_1 becomes an in-run drift control for the single target row.

### Justification
The source run used the current master blob (base_blob 4da9467), so its rows compare directly to BEST. At T=1, Q ran 8.4 us at 4.5 TB/s and O 6.4 us. Full decode_1 measured 139.3 vs BEST 142.9/145.9 and vs the control run's 144.1, so -3.6 to -4.8 us, about 6-8 err. Sliding decode_1 was 125.8 vs control 126.5 (-0.7 us, within err), and the 03:41Z harvest explicitly recommended full only. The patch already has the knobs: DL2_MAX_TOKENS (its own bound, also sizing ws_q/ws_o) and _dl2_applies(), which honours DL2_SLIDING1 != 'raw' to keep sliding decode_1 on master's nvjet + _sliding_forward. Setting DL2_MAX_TOKENS = 1 and DL2_SLIDING1 = 'nvjet' implements the narrowed gate without touching kernel code. One check is readable: a single target row at err ~0.6 us with an expected -3.7 us, plus the dl_qkv_kernel/dl_o_kernel profile rows. With sliding decode_1 off, that row reads batch drift in the same run. Weakest link: the gain is small (~+0.003-0.005 headline) and lives on one shape, so landing will need a submit-grade read either way.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (master): stage 2's O kernel PDL-chains into the stage-1 chain (boundary -> P). The T=1 gain partly depends on P's PDL pre-wait overlap, which is unchanged and landed.
- fa3-hdim128-head-pairing-batched (master): pairing is batch>1 only, so it does not apply at decode_1 (batch 1). Q writes the paired pad layout only where paired, so no interaction at T=1.
- stack-route-topk-pdl-unpool-nf1 (queued sibling): it also improves T=1 (R and unpool). Run both independently on master. The patches conflict in _DL_CUDA (verified: dense-chain-stage2 patch fails to apply after the route patch at ~line 295), so whichever lands second must hand-merge the TU with one module name. Their T=1 gains may be sub-additive through the shared PDL chain.

### Interference and A/B needs
No companion. With DL2_SLIDING1='nvjet' the sliding decode_1 row is master code in the same run and serves as the control. T>1 shapes return to master byte for byte. A separate ab run would only duplicate that.

### Open questions
- BUILDER: implement the gate with the existing constants: DL2_MAX_TOKENS = 1 (also shrinks ws_q/ws_o) and DL2_SLIDING1 = 'nvjet' (full-only at T=1), rename the module, and drop the NF=2/4 instantiations and T>1 self-check cases. Print PATH CHOSEN per T, showing Q/O only at full T=1.
- BUILDER: make the selector compare its printed T=1 timing (Q + FA3 + O vs master nvjet qkv + FusedQKNorm + casts + store_cache + FA3 + FusedRMSNormGate + nvjet o) and assert the custom path wins before enabling it (LESSONS 163).
- If the user prefers to keep sliding T=1 on (the spec as written), the run is still readable on the full decode_1 row, but then there is no in-run decode_1 drift control.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree. The worktree already carried the parent's (dense-chain-stage2-attn-side-pdl) patch, staged; my edits are the unstaged diff (+70/-15).

- Gate (review fix applied, as the reviewer and the spec instructed): `DL2_MAX_TOKENS = 32 -> 1` and `DL2_SLIDING1 = "raw" -> "nvjet"`. `_dl2_applies()` is unchanged: tokens > 1 returns False, and sliding B=1 L=1 returns False under "nvjet". Q and O therefore run only at full-attention decode_1. Sliding decode_1, decode_16/32, verify_* and every other shape run `_master_attention` byte for byte, with the stage-1 chain (P, R, Marlin) unchanged. Sliding decode_1 is the in-run drift control.
- Deviation (necessary): the review says DL2_MAX_TOKENS also shrinks ws_q/ws_o. It cannot: the kernels stride ws by MAXT = 32 and `dl2_qkv`/`dl2_o` TORCH_CHECK `ws.numel() >= SPLIT * MAXT * ROWS`, so a 1-token ws would fail every launch and turn the whole path off. Added `DL2_WS_TOKENS = 32` (must match MAXT) for the ws sizing; ws is 6*32*2304 and 2*32*HIDDEN floats as before.
- Not done, by instruction ("keep the change that small"): NF=2/4 instantiations remain compiled and unused at runtime (T > 1 never reaches them), no kernel code touched, module name left as the parent's `muse_decode_dense_chain_v2` (the parent already renamed it versus master; the review's rename suggestion is skipped to keep the TU identical to the parent).
- Self-check: cases cut to T=1 only: (B1 L1 sliding kv 1.0), (B1 L1 full kv 1.0), (B1 L1 full kv 0.25); graph-replay chain cases now (1,1,full) and (1,1,sliding); `_dl2_time` at T=1 only.
- Review open question applied (LESSONS 163): new `_dl2_compare()` runs after the self-check; it graph-times (20 launches x 5 replays, best of 3, CUDA events) custom Q+O against master's qkv GEMM + FusedQKNorm + casts + store_cache + o_norm_gate + o GEMM on the same T=1 full inputs (FA3 identical on both paths, left out), prints both, and keeps the custom path only if it wins; otherwise or on any exception, PATH CHOSEN = MASTER everywhere.
- PATH CHOSEN lines printed per shape class: full decode_1 (Q+FA3+O), sliding decode_1 (master nvjet, drift control), T > 1 (master).
- Merge note: this patch conflicts with the route patch (stack-route-topk-pdl-unpool-nf1) in `_DL_CUDA` around line 295 (the dl2 declarations in `_DL_CPP` and the TU); whichever lands second must hand-merge the TU and keep one module name.
- Risks: the prepare-time comparison is noisy at ~20 us totals (a false "master wins" would silently turn the gain off; read the log line); no GPU here, nothing was run.

## Rules check

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

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only, no GPU). **Verdict: dispatch** (no edits).

- Gate: `DL2_MAX_TOKENS = 1` + `DL2_SLIDING1 = "nvjet"` makes `_dl2_applies()` true only at full T=1 (B1 L1, 16/2/64 heads). Every other TASK.md shape and test (incl. sliding decode_1, verify_1 T=8, the T=8 ctx 1M test) goes to `_master_attention`, which is master code. Worktree HEAD blob = master blob 4da9467, so the parent's passing check (b31b8453) validates the T=1 full path on these inputs.
- Builder risk 1 (ws sized by `DL2_WS_TOKENS = 32`): correct and required. The kernels index ws by `p * MAXT + t` with MAXT = 32 and TORCH_CHECK `numel >= SPLIT * MAXT * ROWS`; a 1-token ws would fail every launch. Cost is about 1.8 MB.
- Builder risk 2 (`_dl2_compare` noise): no margin or extra reps needed. Each sample averages 100 graph-replayed runs, best of 3, so noise is about 0.1 us. The expected gap is several us (custom 2 kernels, about 15 us at the parent's 8.4 + 6.4 us, vs master's 6+ kernels). The comparison is biased toward custom, not master: `_dl2_ref_q` adds a `qkv.clone()` and a `gate.contiguous()` copy that the real master path does not run. So a "MASTER wins" would mean a real loss, and the realistic failure is an overstated printed gap, not a dropped path. Any exception in compare lands in `_dl2_setup`'s except, which prints `OFF ... PATH CHOSEN = MASTER`. The master() closure is graph-capturable (rope-free full config; ops master already runs under graphs; warmed by the eager self-check).
- Builder risk 3 (self-check at T=1 only): matches the gate. The sliding/raw cases test code paths that never run now. They can only over-gate (turn the path off), and they passed in the parent run.
- Rules: no change to kernel_fn's call graph beyond constants. The compare, its captures and syncs are prepare()-only (lint OK with warnings, all pre-existing or prepare-side). Nothing data-derived is cached, and precision is unchanged. Scope matches the change paragraph; NF=2/4 kept unused, as the spec allows.

Residual risk: the win is one shape at about 2.6%, near check resolution, and the standalone compare does not model the stage-1 PDL overlap; read the full decode_1 row and the `compare T=1 full` log line.
