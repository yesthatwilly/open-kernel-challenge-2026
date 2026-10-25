# Cross-problem lessons

Hand-maintained by churnsubs. Each problem's `FINDINGS.md` is regenerated from its own results.
This file holds what carries over between problems: measured facts about the hardware, the judge
and the SGLang kernels, with the run that showed each one. Read it before `/hypothesize`, and
add to it when a harvest teaches something general. The evidence for every line is in
`<problem>/hypotheses/done/<outcome>/<slug>.{md,result.json}`, and the full okc record and
profile are in `<problem>/hypotheses/archive/<slug>/`.

## Measurement

- Identical code drifts ~1.5-2.5% per shape between runs (decode expert-id-map-single-kernel moved
  every shape +1-2.5% uniformly; qkv_gate decode_1 ranged 0.981-1.025 across five master-equivalent
  bf16 runs). An effect under ~1.5% on the geomean cannot be resolved by one run. Read per-shape
  rows, the paired ratio against SGLang in the same run, and the replaced kernel's own time
  (`top_kernels`), not the headline alone.
- Triton kernels' times reproduce within ~1% between runs; cuBLAS split-K rows swing more.
- Reported shape time minus the profiled kernel sum is a real per-graph-node cost: ~2-4 us for
  nvjet+splitKreduce pairs, ~0.3 us for a single Triton launch (bf16 pooler/unpooler runs).
- Resubmitting byte-identical code gets a new job id (it is not deduplicated) and costs a check.

## Launches and fusion

- Fusing a handful of tiny kernels was flat on decode-sized layer shapes (decode
  cache-store-fused-fp8-cast, expert-id-map-single-kernel, moe align-single-token-ep, prefill
  moe-align-side-stream, all neutral). It pays only where launches dominate the shape:
  attention_decode's fused pre-attention gave 1.127x (dfee6f2939974bad) on 17-36 us shapes.

## GEMMs (bf16, H200)

- The one large bf16 win: the router as one 3-D split-K Triton kernel (grid cdiv(M,BM) x N/BN x
  SPLIT_K, ~128 CTAs, fp32 atomic_add into a zeroed output, BF16 hi+lo split), 10.5-17.7 us to
  5.7-9.3 us at every m, 1.151x (bf16 router-triton-hilo-fused-control-splitk-all-m).
  A full-K grid of only 8 CTAs at m=128/256 was 50-98 us (router-triton-hilo-fused, 0.940x): fill
  the grid.
- Triton full-K weight streams: BLOCK_N=64, BLOCK_K=256, 4 stages (128 KB in flight per CTA)
  reached 3.73 TB/s at m=16 and beat nvjet. BLOCK_N=32 (64 KB in flight) did 3.3 TB/s, and
  BLOCK_N=16 regressed 1.6x (qkv-gate-triton-skinny-m32). At m=32 the penalty is not the 32-row tile:
  two 16-row tiles were worse (unpooler BLOCK_M=16 control).
- Split-K ticket-semaphore reduce (masked partial store, debug_barrier, one gpu-scope acq_rel
  atomic_add ticket, winner reloads .cg, fixed-order sum, atomic_xchg reset) passes correctness
  (bf16 pooler 1.166x and qkv-gate 1.143x vs 1.151x master: both neutral, within drift).
- bf16 router: SPLIT_K 32 at m<=32 was flat with num_stages 2 (1.143x) or 3 (1.152x) vs 1.151x master:
  the router's small-m grid depth is settled.
- bf16 pooler at m<=32: TMA weight loads (1.169x) and BLOCK_K=256 cp.async (1.162x) both landed within
  noise of master 1.151x; TMA adds little over deeper cp.async.
- bf16 m=32: swap-AB (64 weight rows on tl.dot's LHS, 32 tokens on N) emits wgmma (tokens-on-M had
  wgmma=0, 255 regs); pooler m=32 +9.3%, o +4.1%, qkv +3.6%, 1.198x (bf16-m32-swapab-wgmma-triton).
- Several per-shape Triton wins that were each neutral on the headline combine: bf16 shape-gated combo
  (pooler/qkv/o at m<=16, unpooler slice) 1.176x vs 1.151x. Judge single-area runs per shape, then combine.
- A 128x128x64 Triton split-K kernel at m=128 (pooler SPLIT_K 4, unpooler 2) lost to cuBLAS (1.116x vs
  1.151x master; pooler-unpooler-m128-wgmma-splitk-semaphore).
- cuBLAS vs cublasLt backend preference changes nothing on this image (bf16-cublaslt-backend-toggle).
- FMA-broadcast (no tl.dot) GEMV was 10x slower (unpooler-slice-triton-gemv).

## Marlin MoE (W4A16 NVFP4)

- Per-m-block cost grows with block_size_m (thread_m_blocks). The next-larger block lost 18-28%
  (moe marlin-block-size-m-next-larger, 0.918x). Capping at 16 gained 8-15% at T=128/256
  (decode marlin-block-m-cap-16, 1.062x). SGLang's T*8/32 rule overcounts rows per local expert
  8x under EP8.
- On prefill, Marlin block_size_m from local rows per expert lost (0.984x, experts-marlin-block-m-local-rows)
  and block 48 at turn_1 was flat (1.003x, experts-marlin-block-m-48-turn1): block-m tuning there is settled.
- Knobs that did nothing: workspace blocks per SM, use_fp32_reduce=False, use_atomic_add.
- Prefill stage-1 Triton W4A16 grouped GEMM (in-register arithmetic NVFP4 dequant, 128x128x64 tiles,
  SGLang fused_moe_kernel skeleton) is correct but ~2.0 ms for w13+w2 at turn_1 and 3.3 ms at turn_4,
  0.689x overall (experts-triton-w4a16-inkernel-fp4, 218515e2d0b14ce7). Losses shrink with T (0.46x
  turn_1 sliding to 0.90x first_turn), so the cost is per-tile dequant overhead, not the MMA.
- moe small-m-grouped-gemv-splitk-rewrite stage 1 (split-K grouped GEMV) lost 1.7-2.6x at bs 1/8/16 (0.664x).
- Swapping operands so the in-register-dequantized FP4 weight tile is tl.dot's LHS (tokens on the MMA N
  side) plus a lossless nibble reorder in prepare() doubled the Triton W4A16 expert GEMM: w13+w2 went
  2.0 ms -> 0.80 ms at turn_1, 3.3 -> 2.1 ms at turn_4, 13.5 -> 9.1-10.0 ms at first_turn, beating Marlin at
  turn_4/first_turn (prefill experts-triton-w4a16-swapab-reglhs, 1.007x overall).
- Prefill rewrite stage 3: BLOCK_T 256 (8 warps, 2 stages) at T>4096 took first_turn w13+w2 from 9.1-10.0 to
  8.0-8.7 ms (1.023x overall, landed): fewer redundant per-tile dequants at ~512 rows per expert.
- Prefill stage 4: Marlin at T<=4096, Triton NVFP4 above (both banks built in prepare, +0.9 GiB) -> 1.060x:
  route each shape to the engine that measured faster there.
- moe: one Triton map+align launch at bs<=32 gained 4.5-12.6% (triton-map-align-small-m-port) and stacked
  with the bs>=128 block-m pick to 1.312x (triton-map-align-on-block-m-restack). Stacking shape-disjoint wins
  by 3-way merge reproduced both parents' per-shape gains.
- decode_layer: porting moe's single-program Triton post-norm at T<32 gave 1.164x (from 1.087x).
- moe chunked align failed to compile: tl.arange needs a power of 2 and block_size_m 48 is not.
- moe: the block-m win stacked cleanly on the post-norm win: 1.254x (marlin-block-m-two-smaller-on-postnorm).
- moe: smaller Marlin block_size_m at bs=128/256 gained on its own (verify-two-smaller 1.092x,
  next-smaller 1.039x), confirming decode's cap-16 lesson on the standalone MoE.
- moe: replacing SGLang's post-expert norm at bs<32 (where its wrapper launches ~9 at::native
  reduce/elementwise kernels) with one Triton program per token gained 1.130x (post-expert-norm-triton-small-m).
- A Triton W4A16 grouped GEMV with arithmetic FP4 dequant was 2-2.6x slower than Marlin at small M
  (triton-w13-grouped-gemv-small-m). It was dequant-ALU-bound with too few busy CTAs.

## Attention (FA3, FP8 KV, page size 1)

- num_splits is only a cap under varlen dynamic split, and FA3 already sizes splits on the GPU
  (decode fa3-num-splits-* reviews).
- FA3 sits at ~600 TFLOP/s on every full prefill shape whatever the CTA count (fa3-batched-mode review).
- A hand-written CUDA split-KV kernel (load_inline, compiled at import in 41 s) for full-layer decode at batch
  16/32: persistent 2xSM grid over 1024-token chunks, page indices prefetched to smem one item ahead, 4-stage
  cp.async ring of 128-byte K/V rows, f16 mma.sync; plus a combine + norm-gate kernel. Self-check rel L2 1.7e-3;
  attention 1.293x -> 1.347x (full-decode-cuda-gather-splitkv). Triton never got past ~2.1 TB/s on this gather.
- The fused sliding Triton kernel at 16 splits for every decode batch: sliding decode_16 +33%
  (sliding-fused-triton-decode-batched-deep-splits).
- Full-attention decode reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit.
- FA3 head pairing (two hdim-64 KV heads viewed as one hdim-128 row, query zero-padded into its half)
  gained 14-24% on full batch 16/32 decode and verify (attention fa3-hdim128-head-pairing, 1.171x) but lost
  27-35% at batch 1, where the pad/unpad launches dominate. Gate it on batch.
- Writing the padded paired query straight from the pre-attention kernel and un-padding in a Triton
  norm-gate epilogue took attention to 1.244x (paired-heads-launch-cleanup); gating pairing to batch > 1 on
  top gave 1.269x. Ported to decode_layer (batch > 1 only) it gained 7-13% on every full batch-16/32 shape
  (1.087x, fa3-hdim128-head-pairing-batched).
- The fused sliding Triton kernel gated to batch-1 decode only: sliding decode_1 16.6 -> 13.1 us (1.293x).
  Building FA3's schedule on a side stream lost (1.225x, fa3-scheduler-metadata-side-stream).
- One fused sliding-window Triton launch won sliding decode_1 by 24% but lost 13-60% everywhere else,
  2-2.6x on verify (attention sliding-single-launch-fused-attention, 0.945x).
- A Triton split flash-decode reading both KV heads of a token as one paired 128-byte row (decode
  triton-flash-decode-full-paired-kv, e82b11f9d9f64b23) is correct and within ~5-7% of FA3 on full
  decode: 200.8 vs 188.8 us at decode_16, 394.8 vs 374.4 us at decode_32 (1.070x overall, neutral).
- Trimming the sliding FA3 page table to the window on device lost (attention fa3-window-trimmed-page-table,
  1.141x vs 1.269x): FA3 already limits sliding work to the window; the extra launch costs more.
- decode rewrite stage 2 (more splits + explicit one-tile K/V prefetch, num_stages 1) slowed the Triton
  flash-decode to 245/478 us at decode_16/32 from 200.8/394.8 (1.041x): Triton's own num_stages=3
  pipelining was better than the manual prefetch.
- Copying KV into a contiguous buffer per call cost more than it saved on prefill
  (fa3-contiguous-kv-gather, 0.988x).

## Rules rulings (user)

- No per-call BF16 copy of NVFP4 weights (dequant in registers inside the kernel only).
- Never gate a new path on master's `_align_self_check` returning 0: it counts its 'capacity vs SGLang' line as a hard mismatch, and capacity always differs (master logs 60/60). The gate silently always chose the fallback, so the run would have measured master. Caught by a pre-flight audit before dispatch. A pre-flight audit of a CUDA build is worth one agent per check. (moe cuda-nvfp4-swapab-mma-persistent-small-m, 2026-10-24)
- The judge image logs torch 2.14.1+cu130 (muse_bf16_proj archive 5ed9ace10d9a4f68.log), so the CUDA 13 toolchain accepts PTX ISA 8.4+ forms such as mma.sync m16n8k32 e4m3. An FP8 mma.sync build failure is unlikely to be the toolchain. (full-verify-cuda-streamk-splitkv audit, 2026-10-24)
- prefill: the packed PTX NVFP4 dequant (prmt/lop3/mul.bf16x2) logged ON but was flat (expert GEMM 8.0-8.9 ms). The Triton expert GEMM is not dequant-ALU bound, and the asm adds a shared-memory round trip (local_alloc 4->5). (experts-triton-nvfp4-packed-asm-dequant)
- prefill: fusing silu_and_mul into the w13 epilogue removed 287 us of act_and_mul but lost 1-3% on first_turn. (experts-triton-w13-fused-silu-epilogue)
- moe: an L2 prefetch of w2 before Marlin at bs=1 cost 13.8 us and Marlin did not speed up (44.5 us): 0.79x on bs=1. Dead knob. (w2-l2-prefetch-decode1)
- moe: one 33-program Triton align at bs>=128 took align from 10.6/11.4 to 4.0/6.9 us, +3-4% per shape. (align-parallel-per-expert-verify)
- bf16: swap-AB with BLOCK_T=16 (wgmma m64n16) at m<=16 made qkv 18% faster, o 10-19% and the unpooler slice ~12%; pooler flat. (bf16-m16-swapab-wgmma-extend)
- bf16 router: master's hilo kernel already emits wgmma at BLOCK_M=64 (m>=64); swap-AB pays only at m<=32 (+3-5%). (router-swapab-wgmma-all-m)
- bf16: a cuBLASLt algo picked by graph-replay timing in prepare() did not carry over to the judge: qkv m=256 was 6.8% faster in prepare but 14% slower in the judge. Prepare-time algo picking is a dead knob. (bf16-cublaslt-algo-pick-in-prepare-m128-m256)
- decode_layer: the bf16 Triton split-K GEMMs win at T=1 (+3.6%) but lose 2-4.5% at T=8/16 despite faster kernel sums; gate them to T=1. Possibly interference with the side-stream router overlap (unverified). (bf16-shape-gated-triton-gemms-small-t)
- attention: full batch-1 decode through the fused Triton kernel (64 splits) ran at 0.89 TB/s, 5% slower than FA3. FP8 QK mma.sync in the CUDA split-KV kernel was flat, confirming the gather is the limit. (full-fused-triton-decode1-gate, full-decode-cuda-fp8-qk-mma)
- Measurement: an identical CUDA kernel's benchmark median moved 6% between runs (attention full decode_32, 241.9 vs 256-260 us, same profile), and sliding decode_1 drifted +2-3.5% between run batches. Judge restacks on paired_ratio/raw ratios and kernel profiles, not the headline alone. (harvest analysis 2026-10-24)
- A self-check against master's Triton RoPE kernel must not require byte-identical FP8 cache rows: Triton's tl.sin/tl.cos are likely fast approximations, so at large positions (~5000 rad) 1-6% of E4M3 bytes differ from accurate sincosf. Compare by rel L2 against an accurate reference instead. (sliding-cuda-cluster-dsmem-onelaunch audit, 2026-10-24)
- 2026-10-25 `stack-onewarp-hoisted-dispatchcombine-from17` (moe): when stacking, check whether one change's gate is derived from a constant another change edits. The CUDA dispatch path was gated on `ALIGN_FUSED_MAX_TOKENS < T`, so align-parallel-from-17's 32 -> 16 would have silently widened the CUDA path into an un-self-checked range. Give each path its own bound constant.
- 2026-10-25 `experts-cuda-wgmma-mixed-input-grouped-gemm`, `attn-full-prefill-cuda-fp8-exp-offload` (prefill): a torch `load_inline` CUDA build that uses wgmma FAILS on the judge unless it targets sm_90a only: the default arch list adds a plain sm_90 pass, which cannot assemble wgmma, so the run silently measures the fallback. Set `TORCH_CUDA_ARCH_LIST=9.0a` (or pass only `-gencode arch=compute_90a,code=sm_90a`) and guard wgmma code with `__CUDA_ARCH_FEAT_SM90_ALL`. Before trusting any wgmma rewrite's result, grep its log for the build line and PATH CHOSEN.
- 2026-10-25 `cuda-nvfp4-swapab-mma-persistent-small-m` (moe), `moe-chain-cuda-swapab-expert-once-large-t` (decode): a CUDA kernel that hits an illegal instruction during the prepare() self-check poisons the CUDA context, so the in-process try/except fallback cannot save the run (0/0 tests, failed). A child process is NOT an option: the rules statically reject submissions that start processes (CORRECTED 2026-10-25, the adversarial review of moe-chain-cuda-swapab-expert-once-fix caught it). Instead make the first launch unlikely to fault: prefer instruction forms the judge has already run (plain cp.async.cg over createpolicy cache-hint variants), build with -lineinfo so the faulting instruction is named, sync and print after each kernel of the self-check so a fault is attributed, and run such a kernel's first measurement as an `okc check`, never as a scored submit.
- 2026-10-25 `full-verify-wgmma-s3-warpgroup-per-kvhead-inproc` (attention, audit): an FP8 (E4M3) softmax-probability operand P for the PV MMA needs FA3's Max_offset: compute P = exp2(s - (max - 8)) so P lies in [0, 256] before E4M3 rounding (the 2^8 cancels in O/l and in split merges). Without it every weight below ~2^-10 of the row max rounds to 0, and on the judge's coherent long-context data the error broke the tolerance (numpy emulation: rel L2 0.107 vs bound 0.088) while a random-key self-check passed. Test FP8-P kernels on coherent data, not random keys.
- 2026-10-25 `full-decode-splitkv-kv-bw-3p5` (decode): the split-KV kernel-A variant sweep printed 3.04 TB/s at B=16 and 2.81 TB/s at B=32 for every chunk-1024 variant, and 2.73/2.53 TB/s for chunk 512. Every variant ran at 2 CTAs/SM. Stages 4→5 and 1→2 tiles ahead each moved it ±0.3 µs, so ring depth is not the limiter and 3.5 TB/s needs a structural change. The hard-coded selector also picked the slower chunk-512 variant for B≤23 despite its own printed ranking, which cost +18 µs of kernel A at full decode_16. Pick variants from the printed timing, or assert the static choice is the fastest.
- 2026-10-25 harvest: prepare-time CUDA-event kernel timings undercount the judge profile by 20-25%. Split-KV kernel A measured 94.0 µs in prepare vs 112.7 µs in the profile, and 104.5 vs 131.1 µs. Use prepare timings to rank variants, not as the judged number. (full-decode-splitkv-kv-bw-3p5)
- 2026-10-25 `cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc` (moe): the sm_90a-only build fix works for moe too: compiled, 168 regs, TMA descriptors rc 0. But the canary returned `error word 0 (ok), output finite False`, so the path fell back to Marlin and the 1.330x is master re-measured. A finiteness check in the canary is what catches numerical wgmma bugs that do not fault. Make the canary name the first non-finite tensor.
- 2026-10-25 `cuda-nvfp4-swapab-mma-persistent-small-m-inproc` (moe): once the illegal-instruction fault was fixed (probes of cp.async, ldmatrix, cvt e4m3x2 and mma.sync all ok), the swap-AB mma.sync kernel ran at T≤32 and was 1.7-2x slower than Marlin: 75.9 vs 44.7 µs at bs=1, and 415.5 vs 207.5 µs at bs=32. Its self-check rel L2 was 9.2e-3 against a 1e-2 bound. Dead design.
- 2026-10-25 `experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc` (prefill): the sm_90a wgmma grouped NVFP4 GEMM really ran at T=16384, at 14.7-16.2 ms vs Triton's 8.1-9.2 ms (1.8x slower). It had 168 regs with a 680 B local spill at 1 CTA/SM. A wgmma build that runs is not a fast one; check the spill line before spending a check.
- 2026-10-25 `attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc` (prefill): the FP8 wgmma full-prefill attention kernel was correct (better than FA3 vs fp64) but 1.53-1.67x slower than FA3 at every gated shape. The prepare-time hard gate (threshold 0.92x) kept FA3 everywhere at no cost. A hard gate is the safe way to ship an unproven attention kernel to a check.
- 2026-10-25 `bf16-lossless-exp4-occupancy-dequant-once-stage2` (bf16): entropy-coded (12.17 bits/element) bf16 weights stream at only 1.4-1.6 TB/s compressed at 1 CTA/SM. That is 1.7-2.1x slower than master's swapab GEMMs and missed the pooler kill gate (40.3 vs 19.0 µs). The 24% byte saving cannot pay for in-kernel decode. Lossless weight compression is dead.
- 2026-10-25 harvest tooling: `hypotheses/BEST.result.json` is stale in bf16, decode and moe; it holds a reference or baseline job, not the BEST.json job. Compare against the file named in BEST.json `result`. Archived `<job>.log` files are a ~15 KB tail (both moe logs are 15022 B and start mid-line), so print build and PATH CHOSEN lines late or compactly.
- 2026-10-25 `full-decode-splitkv-s2-streamk-ticket-inproc` (attention): stream-K tiles over the page-1 full-decode KV raised split-KV kernel A from 2.79 to 3.05 TB/s (1.10-1.14x from split/locality alone), and the in-kernel ticket combine saved a further 1.03-1.05x vs a separate combine. That is −9.3%/−10.8% kernel time at B=16/32. Stream-K is the structural change the plain ring-depth/chunk sweep could not deliver.
- 2026-10-25 `sliding-cluster-s2-decode-c4-pdl-inproc` (attention): a per-batch-band CUDA kernel can win at some bands and lose at others in the same check (cluster C=4 −17% at B=32, C=8 +7% at B=16). Read each band's kernel row separately and gate the losing band off; do not let the headline average them.
- 2026-10-25 `full-verify-cuda-splitkv-stage2` (decode): the mma.sync split-KV design that beats FA3 on gather-bound full decode is 1.7-1.8x SLOWER than FA3 on q_len-8 verify (64 query rows per kv head is MMA-bound, where FA3's wgmma wins). Do not extend a gather-optimized decode kernel to verify.
- 2026-10-25 `moe-chain-cuda-swapab-expert-once-fix-inproc` (decode): with per-launch `launching`/`ok` prints the 715 did not recur (cache-hint cp.async, fma.bf16x2 −0 and the multi-arch build were all replaced at once, so the cause is unattributed), but the path then turned itself OFF on a self-imposed 1e-2 row bound against Marlin (1.07e-2) while the judge allows 3-15%. Calibrate self-check bounds against an fp32 reference and master's own error, not a fixed number against another approximate kernel.
- 2026-10-25 `cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc` (moe): a prepare-time cold-L2 DIAG that is unimodal (p90/p10 1.03) does not prove the judge's timed loop is: the judge still saw 40% of bs=1 replays in a +23 µs slow mode. Check `runs_ns` in the okc json for bimodality (median vs median_raw vs mean) and instrument the timed instantiation, not a separate one.
- 2026-10-25 harvest: runs on stale bases (missing master's dense chain or a promoted path) shift untouched shapes by 4-15%; read only the changed path's kernel rows from such runs, and rebase before any stage that will be judged on the headline.
- 2026-10-25 (orchestration): a subagent running shell commands from the main repo with an unquoted backtick in an argument executed `git apply -3` against master's tree and left unmerged index entries, which made a concurrent `churn.py commit-impl` fail mid-way. Subagents must quote arguments and run git only inside their own worktree (`git -C <worktree>`).
- 2026-10-25 03:41Z harvest `full-decode1-streamk-s3-two-level-ticket` (attention): at B=1 a single request has too few KV tiles for one-level stream-K. A two-level ticket on an SM-sized grid took full decode_1 from 27.3 to 22.6 µs (FA3 + combine + norm-gate → one 18.6 µs kernel). In prepare it ran 23.1 µs vs 53.9 for stage 2's 2xSM grid. The same kernel is slower at B=16 (115.8 vs 107.0), so gate by batch from the diagnostic.
- 2026-10-25 03:41Z harvest `w2-marlin-atomic-add-small-m` (moe): Marlin `use_atomic_add` on the w2 call alone took bs=1 from 48.8 to 44.9 µs. The judge's +23 µs slow mode fell from 44/100 to 10/100 replays. bs=8/16/32 did not move. Read `runs_ns` mode counts as well as the median; a change can win by removing a mode.
- 2026-10-25 03:41Z harvest `cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix` (moe): a per-bucket diagnostic localised a numerical bug a single finiteness flag could not. N16/N32 token buckets were correct, and w13 at N≥64 and w2 at N=128 gave every row non-finite. Routings that use only small buckets (random) pass every self-check, so self-checks must force each bucket (skewed, all-one-expert) or a bucket bug ships as "ok".
- 2026-10-25 03:41Z harvest (decode, two runs): prepare-time selectors chose a kernel their own timing showed slower. `dl_boundary_kernel` was 3.9 vs 2.8 µs for SGLang's fused kernel, and wide P was 24.9 vs 23.4 µs at T=16. Both were shipped because the gate checked correctness only. Every PATH CHOSEN must compare the printed timings and assert the winner.
- 2026-10-25 03:41Z harvest `bf16-verify-deepgemm-bf16-gemm-nt-route` (bf16): `PATH CHOSEN ... deepgemm` for all 16 (layer, m) pairs, yet every profiled m≥128 kernel was cuBLAS nvjet with master's times. A PATH CHOSEN line proves the selector, not the launched kernel. For third-party libraries, profile one call in prepare and assert the kernel name.
- 2026-10-25 03:41Z harvest `pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite` (bf16): the load_inline build failed with "C++20 or later compatible compiler is required to use PyTorch" because the nvcc flags were only `-gencode=...`. The run silently measured master. Pass `-std=c++20` in `extra_cuda_cflags` whenever you override them.
- 2026-10-25 03:41Z harvest (bf16): attribution `base_result` named the unlanded restack, but all nine bf16 runs executed master's kernels (`_pooler_skinny`, `_router_hilo`). Identify the base from the kernel names in the profile before computing deltas. Small-m untouched rows also drifted +2-6% together in this batch.
- 2026-10-25 03:41Z harvest `experts-triton-persistent-grid-first-turn` (prefill): with 74 live blocks on 132 SMs, the persistent grid was 2.9% (w13) and 4.7% (w2) slower in an interleaved A/B, and the gate kept master. The headline then still read +0.002, and turn_1 rows on untouched Marlin moved −7% to −10% (Marlin profile 644 vs 769 µs). Prefill turn_1 drift is that large, so never read it without a kernel change.
- 2026-10-25 03:41Z harvest (prefill turn 1/4): the Triton NVFP4 expert GEMM loses to Marlin at T=753 (BLOCK_T=32: 1206 vs 769 µs). It also loses at T=3012 even with local-parallel align removing 80% of the padding (2193 vs 2148 µs). Keep Marlin below first-turn sizes.
- 2026-10-25 03:41Z harvest (attention): pairing kv-head rows hurts both Triton sliding decode (kernel 12.9 → 16.8 µs at B=16, plus ~11 µs of host or launch overhead in the timed path) and FA3 sliding verify (main 10.7 → 13.2, combine 4.1 → 7.6 µs). Head pairing is dead on this problem.
- 2026-10-25 harvest `cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix` (moe, job 3bdec0e58d684c17): the judge prepares several state copies (setup record `copies: 2`) and rotates them across repetitions. The module kept its TMA maps in a process-global (`g_maps`) encoded over one state's scratch, and the last prepare() re-pointed it, so the other copy's graph read stale rows from the wrong copy. The run passed every test and reps 0/1, then failed the recheck on rep 2 at rel-L2 1.21. Keep device pointers and descriptors per state, never module-global. Self-check by capturing once, then replaying with refilled inputs while alternating two states; replaying the same inputs proves nothing.
- 2026-10-25 04:36Z harvest `full-verify-wgmma-s4-multitile-fix` (attention): a bisect that adds a fully synchronised A/B variant (cp_wait_all + proxy fence + `__syncthreads` around every tile) separates pipeline races from arithmetic. Here it failed identically, which ruled out the ring and mbarriers. Single-tile segments passed even at ring stage ≥1, and every segment of ≥2 tiles had every row's running max wrong. Suspect any wgmma operand held in registers across loop iterations (the register-A Q fragment was loaded only at segment start) before suspecting the softmax carry. Make the DIAG dump S of tile 1 against the reference, so "S wrong" and "carry wrong" separate in one check.
- 2026-10-25 04:36Z harvest (moe `w2-marlin-atomic-add-verify`, decode `moe-chain-...-s3`): a gate written as "not slower than master by more than X%", or a timing printed as "print only", ships losers. The moe atomic path was +0.3-0.4% slower in its own A/B and went live. The decode CUDA MoE path printed 670 vs 320 µs and went live, costing +37-77% on verify rows. Gates must require the new path to WIN by a margin (≥1.02x, as `ouns` did), not merely to tie.
- 2026-10-25 04:36Z harvest `restack-unpooler-slice-cluster-dsmem-m8` (bf16): a model selector looks like this. It times the incumbent and the new kernel in graph replay with a 256 MB flush, interleaves 41 reps, subtracts an empty-graph baseline, prints both medians and demands ≥1.02x, and it predicted the judge (1.10x in prepare, 1.12-1.14x paired on the judged rows). Copy it for any per-cell kernel swap.
- 2026-10-25 04:36Z harvest (bf16 cluster/DSMEM kernels): on 5-6 µs small-m rows, deleting a graph node can matter more than kernel speed. Router m≤8 (`router_kernel<8,8>`, torch.empty, no memset) took 5.38 to 4.98 µs and 5.47 to 4.93 µs. The unpooler slice m≤8 cluster took 6.15 to 5.34 µs and 6.00 to 5.41 µs. Look for a memset or a second node on every small-m row.
- 2026-10-25 04:36Z harvest `pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild` (bf16): the `-std=c++20` fix made the bulk-TMA build compile and run. The row-owner design is still dead. Its time grows linearly with m (gap over the pure read: +0.6/+6/+25 µs at m=1/16/32, at 1 CTA/SM), and even the pure stream reached only 2.54 TB/s cold. Only m=1 won (−8.3%). A row-owner layout that serialises per-token FMA work behind one CTA per SM only suits m=1.
- 2026-10-25 04:36Z harvest `experts-triton-first-turn-unmasked-loads-l2-hints` (prefill): removing the load masks from the Triton NVFP4 expert GEMM made it 17-18% SLOWER at T=16384 (254 regs and 4 spills either way), and L2 eviction hints added nothing. Masks are not this kernel's cost. Close that knob.
- 2026-10-25 04:36Z harvest (decode): runs on a base without the split-KV still reported headlines at or above master (1.267-1.275 vs 1.268), because SGLang-baseline drift lifted untouched verify rows (paired ratio 1.02-1.07). A stale-base headline near master proves nothing. Rank stale-base runs only on the decode_1 rows (err ≤0.7 µs) against an on-master control run from the same batch.
- 2026-10-25 04:36Z harvest `w2-marlin-atomic-add-t1-port` (decode): the moe Marlin `use_atomic_add` on w2 at T=1 carries over to decode_layer. Sliding decode_1 fell −2.3 µs and full decode_1 −3.5 µs, as a unimodal shift; in decode there was no slow mode to remove. The same flag at T≥33 (moe verify) is flat to slightly slower, so the benefit is specific to M≈1.
- 2026-10-25 04:36Z harvest `stack-streamk-ticket-cluster-c16-c4` (attention): the sliding cluster kernel at C=4 for B=17..32 cut sliding decode_32 from 24.3 to 20.1 µs (−17.6%), and C=16 at B=1 took 13.15 to 12.57 µs. In the B=2..16 band, neither the cluster (C=8, +7%) nor a batched Triton fused kernel (14.6 vs 13.7 µs FA3, decode `sliding-batched-triton-b16-band`) beats FA3. Leave B=2..16 on FA3.
- 2026-10-25 (sliding-verify-sk-pseudo-s2-token-group): an auditor's shell-quoted python edit stripped the string quotes inside an f-string in a fallback-only branch; py_compile and okc_lint both pass on undefined names. After any agent edit, grep the diff for bare names inside f-string braces, and exercise every PATH CHOSEN branch (routed AND not-routed) in a CPU dry run when possible.
- 2026-10-25 prefill experts-turn-rega-s4-port-t4096: the moe swap-AB register-A wgmma NVFP4 grouped kernel (s5 copy-safe) ported to prefill wins big at T=753..4096 (gate 0.65x Marlin at T=753, 0.64x ssdq at T=3012; check 1.105 -> 1.327). Whole-subsystem expert rewrites pay at large T; the prepare-time cold-L2 graph gate matched the judge direction.
- 2026-10-25 prefill experts-turn-rega-s4-first-turn-n128: rega extended to T=16384 (capacity-templated tables, block table via __ldg) beat master's Triton expert block 3520 vs 5583 us (0.63x); first_turn rows 1.16-1.23x, headline 1.327 -> 1.406.
- 2026-10-25 moe cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: deeper weight rings (NMAX32/STAGES10, NMAX64/STAGES7) did not lift B128 (w13 1.83 -> 1.87 TB/s judge-like, 17 touched experts); bytes in flight is NOT the limiter at bs=128 - look at the consumer/dequant. Selector kept B128 on Marlin (0.80-0.81x on zipf a=0.3).
- 2026-10-25 prefill experts-rega-low-class-two-ctas-per-sm: 2 CTAs/SM (setmaxnreg 32/104, 80 regs) on the rega low class gave ~9% (V64 0.887x master on random routing; occ1 diag 0.984x isolates the TLP effect) but the narrower 64-route slab splits popular experts under skewed routing (0.994x), so the worse-of-two gate kept master. TLP helps; it needs the full 128-route slab (smem) to pay under skew.
- 2026-10-25 prefill align-two-phase-all-t-rega: two-phase (count + prefix-place) align replaced _align_parallel's all-ids-per-program scan: 396 -> 23 us at T=16384, 69 -> 9 us at T=3012; bit-identical outputs.
- 2026-10-25 prefill rega inner-loop tries: experts-rega-pingpong-consumers-all-t (5fbaec9a) and -producer-scale-expand-all-t (6be2a8d3) never went live; their own prepare gates measured them SLOWER than master in every class (ping-pong 1.18-1.47x, scale-expand 1.03-1.10x), so their headline/row moves were master re-measured. Row-level noise between master-equivalent scored runs is about +-3-4%: always read PATH CHOSEN/GATE lines before crediting a row.
- 2026-10-25 prefill pre-expert-norm-fused-rega-gather (edf68fa3): rega_gather launched one CTA per capacity slot (8T+480, ~115k at T=16384) though only ~16.6k slots fill; a persistent fused norm+gather gated at 0.33-0.35x of (layer_norm + rega_gather) in all three rega classes.
- 2026-10-25 moe: a row win must be attributed to the path that actually ran. align-two-phase-port-from17's verify_32 +13.4% was the wgmma bucket-gate path (its base predated it); at 256<=T<=512 kernel_fn takes _moe_wgmma, which still runs SGLang's 4-launch align (~11.8 us), so align2 never ran there.
- 2026-10-25 prefill combo-b-normgather-two-cta-low-regate (73b40cd1, 1.4677 vs combo A 1.4605): the 2-CTA low class (grid 264) won 12% on random routing at T=753 (V64 0.878, V32 0.873) but lost on the 1.86x-skew guard (V64 1.035, V32 1.321: chunks grow to 4-8 per expert), so the gate kept master. Combo B therefore ran as combo A, and its +0.5% is within run noise. Two CTAs per expert only pay when per-expert load is flat.
- 2026-10-25 attention sliding-decode1-cluster-per-kvhead (4b608df5, 1.512): per-KV-head clusters lifted sliding decode_1 by 9.9%, with every other row flat. The gain is confined to the row the gate routed, and it stacked onto master with no cross-row cost.
- 2026-10-25 bf16 combos: row-owner reduce-free M=1 (741072dc, 1.2566) gained qkv_gate decode_1 1.16x, pooler decode_1 1.07x and verify_16 1.03-1.09x. The naive stack (0ee20555, 1.2448) carried the same pooler/router wins but lost o decode_16 (0.966), and the cuBLASLt plans added nothing measurable. Wins at M=1 stacked; adding the M128 cuBLASLt cell diluted them.
- 2026-10-25 moe combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 (b395f5b6, 1.3866 vs 1.3752): four small per-class wins stacked on PDL master and every one went live under its own gate: w13 atomic at T=1 0.949, align2 at B256 0.357 (12.5 -> 4.5 us), _align_parallel at T 17..32 0.584, hoisted post-norm at T=32 0.789. Front-end and epilogue wins did not interfere with the PDL w13->w2 chain.
- 2026-10-25 attention c320 combos: chunk320 full decode_1 routed in both runs (0.894 and 0.892 of master's two-level), and tiled pre-attention won only at SV32 (0.964). Every other tiled class gated to master. The naive stack (b9d062aa, 1.5094) and regate (52cbfaeb, 1.5199) were both built before per-kvhead landed, which explains naive's sliding decode_1 loss. Stack c320 + SV32 tiled onto per-kvhead master next.
- 2026-10-25 moe two-stream halves (9a455a2a, 1.3887): splitting the experts into two streams paid only 2-3% at T 8-32 (0.968-0.980). No w13 atomic mode passed its correctness gate in that run, though the same F1 path went live in the toprow combo (b395f5b6) after its tolerance was raised to 1.25e-2. The headline tied toprow within noise.
