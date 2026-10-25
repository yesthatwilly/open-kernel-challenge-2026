---
problem: muse_decode_layer
slug: dense-chain-stage2-attn-side-pdl
created: 2026-10-24T224424Z
base: master
area: qkv GEMM + o GEMM + attention-side small kernels (T<=32)
change: >
  Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage
  plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["dl"] is not None and the new
  state["dl2_ok"] flag; else master byte for byte (T=128/256 control). Add to the existing
  _DL_CUDA TU (rename module muse_decode_dense_chain_v2) two kernels built from the
  dl_pooler_router_kernel<NF> body (same swap-AB mma.sync m16n8k16, 64 weight rows per CTA =
  exactly one head, 4-stage cp.async ring, fixed-order ticket split-K reduce, PDL launch via
  cudaLaunchKernelEx with cp.async of weight stages 0-2 BEFORE griddepcontrol.wait and everything
  that reads activations or writes global AFTER it). KERNEL Q dl_qkv_kernel<NF>: W = dense
  qkv_gate bf16 [2304, 8192] as is (36 head tiles x SPLIT 6 = 216 CTAs). The CTA that draws the
  last ticket of a tile owns the 64 fp32 outputs per token and does the epilogue by tile index:
  tiles 0-15 (q heads): per-head RMSNorm exactly as FusedQKNorm does (read
  sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py for weight/eps, eps RMS_EPS), RoPE when cfg
  rope_theta (neox layout and cos_sin from state["cos_sin"] at data positions, exactly as
  FusedQKNorm applies it), cast to FP8 E4M3 and store to q_attn: unpaired [T,16,64], or when
  master paired predicate holds (window==0 and batch>1 and hkv%2==0) into the own half of the
  zero-padded [T,16,128] row (write 64 zeros to the other half); tiles 16-17 (k heads): same norm
  and RoPE, divide by k_scale (device scalar, only when cfg kv_scale != 1.0, passed as a flag),
  FP8 cast, write bytes to k_cache row cache_loc[t] head h; tiles 18-19 (v): divide by v_scale
  when scaled, FP8 cast, store at cache_loc; tiles 20-35 (gate): bf16 to gate [T,1024]. FA3 call
  unchanged (same q_attn/k_attn/v_attn/metadata logic). KERNEL O dl_o_kernel<NF>: W = dense o bf16
  [8192, 1024], 128 row tiles x SPLIT 2 (K halves of 512 = 8 heads) = 256 CTAs at 3/SM (one wave);
  prologue after griddepcontrol.wait loads FA3 out (bf16, unpaired [T,16,64] or paired [T,16,128]
  taking its own half) for its 8 heads, per-head scaleless RMSNorm eps RMS_EPS times sigmoid(gate)
  in fp32 exactly as FusedRMSNormGate, writes the bf16 x tile into the smem x ring (all K of the
  half fits: 32 x 512 x 2 B = 32 KB), then the weight GEMV; epilogue writes attn_output bf16
  [T,8192]. kernel_fn on the gate: qkv = None; Q replaces _gemm(attn_input, qkv_gate), qk_norm,
  the k/v divide, both .to(FP8), store_cache and the q cast / pad copies; O replaces the unpad
  copies, o_norm_gate and _gemm(o, dense o). attn_boundary, P, R, experts, unpooler, ffn_boundary
  unchanged. PREPARE self-check (synthetic, T in 1, 8, 9, 16, 32; sliding+rope and full; paired
  and unpaired; kv_scale 1 and 0.25): q bytes vs master path rel L2 <= 1e-2 (decode fp8 back to
  f32; do not require byte identity, LESSONS: Triton sin/cos), cache rows rel L2 <= 1e-2 at
  cache_loc and untouched elsewhere, attn_output rel L2 <= 5e-3, plus a CUDA-graph capture+replay
  of the whole gated chain; any failure sets dl2_ok False (master attention side, stage-1 FFN
  kept). Print [dense-chain-s2] ON|OFF pdl=.. max errs, and Q/O CUDA-event times pdl 0/1 for T=1
  and 32.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent 132.7/142.6 us: qkv nvjet 13.1 + o 7.0 + 2 splitKreduce ~3.4 + qk_norm/casts/store/q cast/norm-gate ~6 nodes ~10 us -> Q ~11 (37.7 MB at ~3.5 TB/s) + O ~5.5: ~12-15 us off (+9-11%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "same chain at T=8, 186.5/195.9 us: ~11-14 us off (+6-7%)"
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "paired pad/unpad elementwise rows 6.5+4.5 us vanish plus qkv 12.7 us splitK and norm nodes: ~15-18 us off of 347.5 (+4-5%)"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: ~10-12 us off 217 (+5%)
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~10-16 us off 247/535 (+2-6%). Headline ~+3-5% if Q/O stream >=3.4 TB/s
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "T=128 outside gate: control"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "T=256 outside gate: control"
risk: high
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "T<=32 attention side: qkv nvjet GEMM + splitKreduce, FusedQKNorm, k/v divide + FP8 casts, store_cache, q FP8 cast or paired pad copies, paired unpad copies, FusedRMSNormGate, o nvjet GEMM"
review: proceed
status: running
review_at: 2026-10-25T003616Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T003624Z
base_sha: 6a839f514cc7024729ac75ac28402e894091c0bd
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl
worktree: .worktrees/muse_decode_layer/dense-chain-stage2-attn-side-pdl
mode: null
restack_of: null
stale_base: null
---

# dense-chain-stage2-attn-side-pdl

## Change

Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["dl"] is not None and the new state["dl2_ok"] flag; else master byte for byte (T=128/256 control). Add to the existing _DL_CUDA TU (rename module muse_decode_dense_chain_v2) two kernels built from the dl_pooler_router_kernel<NF> body (same swap-AB mma.sync m16n8k16, 64 weight rows per CTA = exactly one head, 4-stage cp.async ring, fixed-order ticket split-K reduce, PDL launch via cudaLaunchKernelEx with cp.async of weight stages 0-2 BEFORE griddepcontrol.wait and everything that reads activations or writes global AFTER it). KERNEL Q dl_qkv_kernel<NF>: W = dense qkv_gate bf16 [2304, 8192] as is (36 head tiles x SPLIT 6 = 216 CTAs). The CTA that draws the last ticket of a tile owns the 64 fp32 outputs per token and does the epilogue by tile index: tiles 0-15 (q heads): per-head RMSNorm exactly as FusedQKNorm does (read sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py for weight/eps, eps RMS_EPS), RoPE when cfg rope_theta (neox layout and cos_sin from state["cos_sin"] at data positions, exactly as FusedQKNorm applies it), cast to FP8 E4M3 and store to q_attn: unpaired [T,16,64], or when master paired predicate holds (window==0 and batch>1 and hkv%2==0) into the own half of the zero-padded [T,16,128] row (write 64 zeros to the other half); tiles 16-17 (k heads): same norm and RoPE, divide by k_scale (device scalar, only when cfg kv_scale != 1.0, passed as a flag), FP8 cast, write bytes to k_cache row cache_loc[t] head h; tiles 18-19 (v): divide by v_scale when scaled, FP8 cast, store at cache_loc; tiles 20-35 (gate): bf16 to gate [T,1024]. FA3 call unchanged (same q_attn/k_attn/v_attn/metadata logic). KERNEL O dl_o_kernel<NF>: W = dense o bf16 [8192, 1024], 128 row tiles x SPLIT 2 (K halves of 512 = 8 heads) = 256 CTAs at 3/SM (one wave); prologue after griddepcontrol.wait loads FA3 out (bf16, unpaired [T,16,64] or paired [T,16,128] taking its own half) for its 8 heads, per-head scaleless RMSNorm eps RMS_EPS times sigmoid(gate) in fp32 exactly as FusedRMSNormGate, writes the bf16 x tile into the smem x ring (all K of the half fits: 32 x 512 x 2 B = 32 KB), then the weight GEMV; epilogue writes attn_output bf16 [T,8192]. kernel_fn on the gate: qkv = None; Q replaces _gemm(attn_input, qkv_gate), qk_norm, the k/v divide, both .to(FP8), store_cache and the q cast / pad copies; O replaces the unpad copies, o_norm_gate and _gemm(o, dense o). attn_boundary, P, R, experts, unpooler, ffn_boundary unchanged. PREPARE self-check (synthetic, T in 1, 8, 9, 16, 32; sliding+rope and full; paired and unpaired; kv_scale 1 and 0.25): q bytes vs master path rel L2 <= 1e-2 (decode fp8 back to f32; do not require byte identity, LESSONS: Triton sin/cos), cache rows rel L2 <= 1e-2 at cache_loc and untouched elsewhere, attn_output rel L2 <= 5e-3, plus a CUDA-graph capture+replay of the whole gated chain; any failure sets dl2_ok False (master attention side, stage-1 FFN kept). Print [dense-chain-s2] ON|OFF pdl=.. max errs, and Q/O CUDA-event times pdl 0/1 for T=1 and 32.

## Expected

- `*load=decode_1 *` faster: incumbent 132.7/142.6 us: qkv nvjet 13.1 + o 7.0 + 2 splitKreduce ~3.4 + qk_norm/casts/store/q cast/norm-gate ~6 nodes ~10 us -> Q ~11 (37.7 MB at ~3.5 TB/s) + O ~5.5: ~12-15 us off (+9-11%)
- `*load=verify_1 *` faster: same chain at T=8, 186.5/195.9 us: ~11-14 us off (+6-7%)
- `*layer=full load=decode_16 *` faster: paired pad/unpad elementwise rows 6.5+4.5 us vanish plus qkv 12.7 us splitK and norm nodes: ~15-18 us off of 347.5 (+4-5%)
- `*layer=sliding load=decode_16 *` faster: ~10-12 us off 217 (+5%)
- `*load=decode_32 *` faster: ~10-16 us off 247/535 (+2-6%). Headline ~+3-5% if Q/O stream >=3.4 TB/s
- `*load=verify_16 *` flat: T=128 outside gate: control
- `*load=verify_32 *` flat: T=256 outside gate: control

## Risk

High: Q epilogue must match FusedQKNorm (weights, RoPE layout) and the cache byte layout; a wrong RoPE fails tests, so the self-check gates it. LESSONS: small-kernel fusion alone has been flat, so the gain must come from the dense streams plus node count together; judge on replaced rows. Overlaps queued fused-preattention-paired-pad-restack and held prefork-qkv-o-swapab-t16-t32 (same qkv/o rows at T=16/32 with Triton); if either lands first, compare against master at claim. The running sliding-fused-batched-decode-port replaces the sliding batch>1 attention chain: if it lands first, keep its branch on sliding batch>1 and gate this to the remaining shapes.

## Milestones

Rewrite stage 2 of `T<=32 attention side: qkv nvjet GEMM + splitKreduce, FusedQKNorm, k/v divide + FP8 casts, store_cache, q FP8 cast or paired pad copies, paired unpad copies, FusedRMSNormGate, o nvjet GEMM`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T003616Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is sound and the target is physically reachable: 37.7 MB of qkv plus 16.8 MB of o weight, about 15 us at 3.6 TB/s against roughly 20-24 us of incumbent GEMM rows plus small nodes. But the spec predates the landed sliding batch-1 fused path (_sliding_forward), so at sliding decode_1 it must not touch the attention chain, and its expected gain there is overstated. Proceed with no companions. The builder must apply the sliding decode_1 carve-out, use its own token-bound constant, and add a kv_scale=1/64 self-check case.

### Justification
Q and O are swap-AB mma.sync split-K GEMVs from the dl_pooler_router_kernel body, which already ran on the judge (no faults) at 3.6 TB/s at T=1 and 2.85 TB/s at T=32. Q is 36 head tiles x SPLIT 6 = 216 CTAs, the same wave shape as P, so Q should land near 10.5-11 us at T<=8 against qkv nvjet 13.2 us plus splitKreduce. O is 16.8 MB, ~4.7 us plus a norm-gate prologue, against nvjet o ~7.0 us plus reduce ~1.7 us. GEMV-only saving ~4-6 us per T<=8 shape; node removal adds ~6-10 us; M3 (<=1.5x floor) reachable at T<=8. Weak links: (1) the expected rows predate stack-align-par-sidestream-sliding1 (master 8fa8fbf); at sliding decode_1 _sliding_forward reads raw [T,2304] qkv and returns normed-gated o, so only the qkv nvjet and the o GEMM are replaceable there: ~5-7 us off 127.5 us (+4-5%), not +9-11%. Incumbent decode_1 is now 127.5/142.9 us. (2) At T=16/32 Q inherits P's x L2 re-read (2.85 TB/s), so the decode_32 gain rests mostly on node removal, which LESSONS found flat alone, except the full decode_16/32 paired pad/unpad rows (~11 us). PDL buys ~0 here: no two of our kernels are adjacent; do not count PDL in the milestone read. verify_16/32 flat is the in-run control.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1 (master): _sliding_forward owns sliding decode_1 from qkv output to normed-gated o. On that shape the builder must either (a) run Q in raw mode (bf16 qkv [T,2304] for _sliding_forward) and O without the prologue, or (b) keep nvjet there; config-only, printed in the log.
- small-t-launch-minimal-dense-chain-pdl (parent, master): do not reuse DL_MAX_TOKENS; use DL2_MAX_TOKENS = 32 (LESSONS 2026-10-25: each path its own bound constant). A moved P row would be a cache/L2 or code-layout interaction.
- fa3-hdim128-head-pairing-batched (master): at full decode_16/32 Q writes zero-padded [T,16,128] paired q and O reads its half of FA3's paired output; report pad/unpad removal (~6.5 + 4.5 us) separately from Q/O stream rates.
- full-decode-splitkv-restack-on-pdl (in flight): if it lands before the claim, O's prologue would duplicate the splitkv combine's norm+gate at full decode_16/32; re-gate O and re-derive expected rows.

### Interference and A/B needs
No companion warranted. A rewrite judged on replaced-row time; the profile separates dl_qkv_kernel and dl_o_kernel from incumbent nvjet rows, and prepare prints Q/O CUDA-event times at pdl 0/1 for T=1 and T=32. Judge on the Q/O rows plus the count of vanished nodes, with verify_16/32 as the flat control. Marlin rows are noise.

### Open questions
- Sliding decode_1: Q raw mode + O without prologue feeding _sliding_forward (preferred), or keep nvjet?
- Does the nvjet_sm90_tst_64x16/64x32 row at decode_16/32 aggregate qkv + o + the unpooler slice?
- Will Q at T=32 hit P's 2.85 TB/s ceiling? If so stage 3 should share dl-p-wide-tile-t16-t32's BLOCK_N 128 tiling.
- Does FA3 here execute griddepcontrol.launch_dependents? Print O's pdl 0 vs 1 event times.

