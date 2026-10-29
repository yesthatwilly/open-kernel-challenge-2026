# muse_attention_decode · 2026-10-29T160833Z · check · fa3-metadata-fold-preattn-full-verify

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-29T154210Z-fa3-metadata-fold-preattn-full-verify.md` (branch `hyp/muse_attention_decode/fa3-metadata-fold-preattn-full-verify`).

- **Job:** `6d1082a6c0dd4f80` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `8de6524e`

## Goal
Fold FA3's varlen scheduler metadata (what flash::prepare_varlen_num_blocks_kernel<NumWarps, Sort> writes, 2.8-3.4 us per call on the board profile) into master's existing _pre_attention_kernel launch on the three full-layer verify rows (window 0, q_len 8, page 1, tp8, B = 1 unpaired and B = 16 / 32 paired), and pass the result as scheduler_metadata= to flash_attn_with_kvcache so FA3 skips its in-call prepare node (params.skip_scheduler_metadata_computation). Nothing is cached: the metadata is recomputed from this call's cache_seqlens on every call, into a fresh torch.empty int32 buffer. Only shape-only constants are frozen at import. (1) LAYOUT, for this build (the profile names prepare_varlen_num_blocks_kernel<1|2, true>, i.e. upstream hopper/flash_prepare_scheduler.cu with batch sort; causal non-local so varlen_sort_batches and head_swizzle are both on): an int32 vector of 4*b_rounded + 1 (b_rounded = round_up(B, 4)) laid out as [0, b_r) num_splits_dynamic, [b_r, 2b_r) num_m_blocks, [2b_r, 3b_r) varlen_batch_idx, [3b_r, 4b_r) num_nheads_in_l2, [4b_r] tile_count_semaphore. Confirm the size at import from the library's own get_scheduler_metadata output (the shape is checked by mha_fwd's CHECK_SHAPE). (2) SEMANTICS to port (one program, B <= 32): n_blocks = ceil(cache_seqlens[b] / BLOCK_N), m_blocks = ceil(L * (QHPK if PACKGQA else 1) / BLOCK_M); total = sum over b of m_blocks * n_blocks; blocks_per_sm = int(ceilf(float(total) * 1.1f * float(NUM_HEAD) / float(NUM_SM))) in fp32 exactly; splits = max(min(ceil(n_blocks / blocks_per_sm), NUM_SPLITS_STATIC), 1) (1 when NUM_SPLITS_STATIC == 1); num_nheads_in_l2 = 16/8/4/2/1 by the first n_blocks * k <= MAX_KVBLOCKS_IN_L2 (with the n_blocks the upstream code uses at that point), times QHPK when not PACKGQA, capped at NUM_HEAD; sort key (causal) = n_blocks * BLOCK_N - m_blocks * BLOCK_M; batches are placed in descending key order, stable on batch index (cub BlockMergeSort is a merge sort; implement it as rank_b = #{j: key_j > key_b} + #{j < b: key_j == key_b} over a [32, 32] compare tile, no tl.sort) and each vector is written at the sorted (virtual) index, with varlen_batch_idx[rank_b] = b; pad slots [B, b_r) of each vector are written 0; the semaphore is written 0 every call (the prepare kernel zeroes it; with skip=true nothing else does, so this is required for graph replay). Port the exact statement order from the upstream file the profile names (where n_blocks is reduced by splits relative to the nheads_in_l2 and key computation) rather than from this paragraph; step (4) is the arbiter. (3) WHERE: add constexpr META (default False, so every other route's compile is unchanged) plus the constexprs above and pointers seqlens_ptr / meta_ptr to _pre_attention_kernel; launch it with grid (tokens + 1, hq + hkv) on the routed rows; program (t == tokens, h == 0) computes and stores the metadata with 1 warp, programs (t == tokens, h > 0) return at once, every other program is unchanged. In kernel_fn, under a config-and-shape-only predicate _fm_gate(cfg, page_table.shape[1]) that reads an import-frozen MappingProxyType _FM_ROUTE keyed by (B, max_pages) -> the constexpr tuple, allocate meta = torch.empty(4*b_r + 1, int32), launch the pre-attention kernel with META=True, and pass metadata = meta to FA3 in place of None (verify_1 unpaired today passes data["scheduler_metadata"], which is None on verify; B = 16 / 32 paired pass None). Nothing else changes: q layout, pairing, window/causal args, descale, epilogues. (4) IMPORT GATE (_fm_setup, per row B in {1, 16, 32} at the judge's shape: ctx 80000 coherent, its page-table width): (a) recover the shape-only constants (BLOCK_M, BLOCK_N, PACKGQA, NUM_HEAD = h_k-of-call if PACKGQA else hq, NUM_SPLITS_STATIC, NUM_SM = multiProcessorCount, MAX_KVBLOCKS_IN_L2 = 8 MiB / (BLOCK_N * (hd + hd) * 1)) by matching a torch emulation of (2) against sgl_kernel.flash_attn.get_scheduler_metadata called with exactly the arguments mha_fwd derives for this call (as fa3-scheduler-metadata-side-stream did: num_heads_k and headdim of the actual paired / unpaired views, max_seqlen_k = page_table.shape[1] * page_size, causal=True, window (-1, -1) after mha_fwd's clamps, qkv_dtype e4m3, num_splits=0) over >= 64 draws of the judge's length distribution plus edge sets (all-equal lengths for sort ties, lengths below BLOCK_N, lengths == L, one long + many short); every candidate tuple that matches all samples must agree on a 64-draw holdout, else the row is not routed. (b) The Triton META program's output equals the library's on every valid slot ([0, B) of each vector, and the semaphore) for all samples. (c) FA3 output and the full kernel_fn output with the folded metadata are torch.equal to master's (metadata None) on >= 8 draws, and two consecutive replays of one captured graph are equal (semaphore reset). (d) Race: alternating CUDA-graph replays of master's kernel_fn step vs the fold on cloned judge-shape inputs, two 31-rep rounds; route when both rounds are >= 1.02x at B = 1 and >= 1.005x at B = 16 / 32 with the IQR bound master's gates use; else that row stays master. Print one PATH CHOSEN line per row with the measured step times, the prepare-node delta and why. Fallback everywhere = master unchanged.

## What changed, and where
**Mechanism:** FA3's varlen scheduler metadata (prepare_varlen_num_blocks<NumWarps, Sort=true>) computed per call by one extra META program of master's _pre_attention_kernel launch (grid tokens+1) and passed as scheduler_metadata, removing FA3's in-call prepare node on full verify B=1 (unpaired) / B=16, 32 (paired), import-gated per row.

Built in the worktree's `muse_attention_decode/submission.py` only (+620 / -3). No commit, no okc run.

- **Kernel (`_pre_attention_kernel`, ~L150).** The unchanged per-(token, head) body moved verbatim into the inlined `@triton.jit _pre_attention_body(..., t, h, ...)`. The kernel gained trailing defaulted params `seqlens_ptr=None, cuq_ptr=None, meta_ptr=None` and constexprs `META=False, FM_T, FM_B, FM_P, FM_BR, FM_BM, FM_BN, FM_PG, FM_QHPK, FM_NH, FM_CAP, FM_NSM, FM_L2, FM_DIV, FM_TIE`, so none of the ~10 existing call sites change.
  - `META=False` is a static branch that compiles exactly the body.
  - `META=True` puts an explicit runtime branch around the whole body (review fix 2, no bare early return): `if t == FM_T: (if h == 0: _fm_metadata(...)) else: body`.
- **`_fm_metadata` (new `@triton.jit`, ~L82).** One program and one warp; it reads this call's `cache_seqlens` and `cu_seqlens_q` (fix 3).
  - `n_blocks = ceil(len/BN)`; `m_blocks = ceil(lq*(QHPK if PG)/BM)`.
  - `blocks_per_sm = ceilf(fp32(total)*1.1f*NH/NSM)`, computed in the same fp32 order, with the ceil done in integers.
  - `splits = clamp(ceil(nb/bps), 1, CAP)` (fix 1 clamp).
  - In upstream order, n_blocks is divided by splits before both the causal key `nbs*BN - mb*BM` and `nheads_in_l2` (16/8/4/2/1 vs L2, times QHPK if not PG, capped at NH).
  - Descending rank comes from a [P,P] compare tile, with the tie rule as a constexpr (TIE 0 stable / 1 reversed). Each vector is stored at the sorted index, with `varlen_batch_idx[rank]=b`.
  - Pad slots [B, BR) are written 0, and the semaphore at [4BR] is written 0 on every call (fix 3).
- **`kernel_fn` (~L1973 to L2013).**
  - `fm = _fm_gate(cfg, page_table.shape[1])` reads the import-frozen `MappingProxyType _FM_ROUTE` keyed by (B, width). It checks only config and the shape.
  - On a route: `fmeta = torch.empty(4*BR+1, int32)` is allocated per call (fix 3), `_fm_pre_launch` runs (grid `(tokens+1, hq+hkv)`), and then `metadata = fmeta` goes into the unchanged FA3 call.
  - Nothing else changed: q layout, pairing, window/causal, descale and epilogues are as before.
- **Import gate `_fm_setup` (new section before the `if torch.cuda.is_available()` block; called last, after `_sw_setup`).** Rows are B=1 at width 80008 (spread fixed) and B=16/32 at 86008 (agentic), the judge's verify widths at ctx 80000. Each row starts only if the import clock is <= 480 s. A row is not routed on any miss or exception.
  - **(a) `_fm_recover` plus holdout.** This step recovers the shape-only constants. It runs a Python port (`_fm_emulate`, with fp32 emulated exactly via struct) over BM {64,128,192} x BN {64..256} x PG x DIV x TIE.
    - The reference is `get_scheduler_metadata` called with the FA3 call's exact args (fix 5): num_heads 16, num_heads_k/headdim of the actual view (1/128 paired, 2/64 unpaired), `cu_seqlens_q`, `max_seqlen_k = width*1`, causal, e4m3, num_splits 0.
    - Samples are 64 judge draws plus edge sets: all-equal lengths (tie rule, fix 4), lengths == q_len (8), below/at/above one block, ramps, "one long + many short" at three positions, "one long + all 8", half long/half short, and uniform. At B=1 there are also many non-80000 lengths (fix 3).
    - Split cap (fix 1): it is recovered only from cap-binding requests (uncapped > library split). All binding values must agree. With none binding, it is the smallest consistent cap (the max split seen), never larger.
    - The library output must have size 4*BR+1, a permutation in the idx vector and a 0 semaphore. Otherwise the layout differs and the row is not routed.
    - Every surviving candidate must also match a fresh 64-draw holdout.
    - A pure-Python self-test of this logic, against a synthetic library with hidden constants, recovered BN and a binding cap exactly at B=1/16/32 (57/8/4), clamped to 108 (< true 200) when nothing binds, and discriminated the tie rule at B>1.
  - **(b) `_fm_triton_meta`.** The production META launch (same grid and constexprs, small dummy token inputs, buffer poisoned with -7) must equal the library on every valid slot and the semaphore, with pads 0, for all recovery and holdout samples.
  - **(c) `_fm_equal_check`.** This runs 8 judge-shape draws (B=1: 4 at 80000 and 4 at other lengths).
    - kernel_fn with the fold vs master (route off) on cloned inputs must give torch.equal output and torch.equal **entire** k_cache and v_cache (fix 2).
    - FA3 output with the folded metadata must be torch.equal to metadata None.
    - The folded metadata must equal the library at the judge shape.
    - Then one captured graph is replayed twice (equal to each other and to master: semaphore reset), refilled in place with other lengths (B=1: 54321) and replayed again (equal to master on the refilled inputs, k/v equal).
  - **(d) Race.** Pre-attention alone runs with META off vs on (15 reps, fix 6). Then two interleaved 31-rep rounds of CUDA-graph replays compare master's kernel_fn step vs the fold, on separate clones of judge-shape inputs, with a cold L2 and the empty graph subtracted (`_sw_rounds` / `_sw_stats`).
    - Bars (fix 7): B=1 >= 1.02x and B=16/32 >= 1.00x in both rounds, plus master's IQR bound (fold IQR <= 1.5 x max(master IQR, 0.05)).
  - **Output.** One `PATH CHOSEN: full verify B=<b> [fm ...]` line per row, compact so the 400-char re-print keeps it (fix 6): metadata-match (with cap binding/clamped), k/v-cache equal, race ratios with the step times and the derived prepare-node delta ((master - fold) + (META on - off)), pre-attn META off/on, and the reason. The full detail goes to a `[fm] B=<b>:` line.
- **Deviations from the spec.**
  1. The candidate grid adds DIV (whether n_blocks is divided by splits before the key and nheads) and TIE as dimensions. Gate (a) arbitrates; on the synthetic self-test DIV=1 is discriminated at B>1.
  2. META reads `cu_seqlens_q` for m_blocks, as upstream does, instead of hard-coding L=8.
  3. META constexprs ride on `_pre_attention_kernel` as defaulted kwargs instead of positional params, so existing call sites are untouched.
  4. The B=1 holdout and recovery sets include uniform lengths, because the judge's B=1 draws are all 80000.
  5. The route also requires kv_scale == 1.0, which is conservative: the scored rows are 1.
  6. `_FM_BAR` is 1.00 at B=16/32, per review fix 7, instead of the md's 1.005.
- **Uncertainties (static review only; nothing was run).**
  1. Triton fp32 `/` lowers to div.full.f32. FA3 is presumably built with --use_fast_math. A blocks_per_sm ceil could differ near an integer in rare cases, but gate (b) checks every sample. A rare live mismatch would still produce valid metadata (splits <= the recovered cap, so memory-safe, and the idx vector is still a permutation), just not bitwise master.
  2. cub BlockMergeSort's tie order is covered by the TIE dimension.
  3. Triton 3.8 must accept `=None` pointer defaults and jit-helper calls with constexpr args; both are standard in 3.x.
  4. The setup runs last (after sw); if the import clock is past 480 s, every row stays master.
  5. Import cost is estimated at roughly 10-20 s per row.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/fa3-metadata-fold-preattn-full-verify diff` (1 file, +620/-3).

- [x] CUDA-graph capturable: kernel_fn's new code calls `_fm_gate` (dict lookup on config plus `page_table.shape[1]`), allocates `torch.empty(4*BR+1)` (sized by shape), and launches one Triton kernel. There is no `.item()`/`.cpu()`/`synchronize()` or tensor print in kernel_fn's call graph (lint reports none there), and no control flow on input values. Gate (c) captures and replays kernel_fn with the route on.
- [x] Nothing derived from `data` is cached across calls: `_FM_ROUTE` holds only shape-only constants recovered from synthetic import-time inputs, keyed by (B, page-table width). The metadata is recomputed from each call's `cache_seqlens` into a fresh per-call buffer, and the semaphore is re-zeroed every call.
- [x] Weights: n/a: this problem has no weights; nothing is repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: unchanged. The pre-attention body is byte-identical code, and FA3's q/k/v and the epilogues are untouched. Gate (c) requires bitwise-equal output and caches vs master.
- [x] Attention sees every key: FA3 still attends over the full context. Only its tile schedule is supplied, computed exactly as FA3's own prepare kernel would compute it (gate c: FA3 output torch.equal to metadata None).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new imports are stdlib `math` and `struct`. `get_scheduler_metadata` is imported from sgl_kernel inside the import-time gate only. `_FM_ROUTE` is this module's own global.
- [x] All work happens inside the timed call; no output caching or replay: the metadata program runs inside kernel_fn's pre-attention launch on every call. Nothing is precomputed outside the timed call.
- [x] First line `# okc: problem=muse_attention_decode` kept; no `triton.autotune`.

`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode` (exit 0; 109 warnings, 0 errors, all "outside kernel_fn's call graph"; the new ones are the [fm] gate's import-time syncs at L9626-L9887):

```
warn  host-sync    L2171: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2174: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2179: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2199: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2206: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2211: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2272: left = int(_sk_ticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2290: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2324: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2335: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2445: left = int(_sk_ticket(dev).abs().sum().item()) + int(_sk_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2472: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2488: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2753: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2757: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2785: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2789: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2797: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2853: left = int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2869: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2894: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2906: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2972: left = int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3029: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3037: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3047: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3051: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3070: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3074: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3088: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3104: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3121: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3668: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3671: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3674: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3737: left = int(_sk_ticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3753: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3756: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3855: left = int(_sk_ticket(dev).abs().sum().item()) + int(_ps_ticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5341: pt = data["page_table"].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5342: seqs = [int(x) for x in data["seq_lens"].cpu()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5351: loc = data["cache_loc"].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5367: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5409: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5729: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5733: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5792: left = int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5807: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5811: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5828: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5830: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5846: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5876: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5884: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5894: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5904: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5989: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5994: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6009: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6031: left = int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6083: left = int(_ps_ticket(dev).abs().sum().item()) + int(_ps_gticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7016: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7029: ndiff = int((a[loc] != b[loc]).sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7058: seq = int(csl[0].item()) if ok else -1  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7059: loc = int(loc_t[0].item()) if ok else -1  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7062: and int(data["positions"].reshape(-1)[0].item()) == seq - 1 and int(pt[0, seq - 1].item()) == loc  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7063: and bool(((pt[0, :seq] >= 0) & (pt[0, :seq] < rows)).all().item()))  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7085: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7088: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7151: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7159: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7169: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7174: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7196: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7212: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7652: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7654: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7670: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L7750: q = torch.full((tokens, hq, width), 0x7F, dtype=torch.uint8, device=dev).view(FP8)  (allowed only where SGLang does the same)
warn  host-sync    L7755: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9052: return int(_sw_buf(device)[-1].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9057: return not bool(_sw_buf(device)[:128].any().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9062: a = _SW_EXT.sw_verify_attrs().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9088: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9097: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9102: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9106: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9123: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9130: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9208: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9221: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9249: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9252: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9269: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9379: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9626: return m.cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9756: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9757: return meta.cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9840: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9847: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9850: if not _fm_same(meta.cpu().tolist(), _fm_lib_meta(lib, [n + 8 for n in lens], mp, B > 1, dev), B, BR):  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9862: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9866: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9868: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9871: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9874: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9884: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L9887: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_attention_decode/submission.py
```

`python3 -I -m py_compile` OK; no `__pycache__` left in the worktree.

## Risks / what could fail
medium
User-requested under 'run all you want that youre able to' (2026-10-29 1504Z); below the value bar: ~0.73% headline is under the judge margin (1.24%), so it cannot land alone and is read per row (prepare row gone, full verify rows -2 us) or stacked later. Correctness risk: the port must match this build's prepare_varlen_num_blocks_kernel<NumWarps, Sort=true> bit for bit (statement order, fp32 ceilf, stable descending sort, nheads_in_l2 rule, semaphore zero); the import gate compares against the library's own get_scheduler_metadata and FA3 output torch.equal, and falls back to master on any mismatch, so the failure mode is 'no route' (flat), not wrong output. Version risk: if the installed build differs from upstream (vector count, layout, pad semantics), gate (a) finds no matching tuple and nothing routes. Perf risk: the extra program (one warp, B <= 32, [32,32] rank tile) must not lengthen the pre-attention launch by more than ~0.3 us; the race catches it. The side-stream attempt showed full verify_1 losing 0.7 us when the prepare kernel was moved rather than removed; here no kernel and no stream edge is added. Not input-derived caching: only shape-only constants (block sizes, static split cap, SM count) are frozen per (B, page-table width); metadata is rebuilt every call.

## Result (after the run)
- job `6d1082a6c0dd4f80` (check), status done, passed True, tests 16/16
- speedup 1.6336x, 95% CI [1.6091, 1.6664], score 2305.5329268084297 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/BEST.result.json
- **outcome: neutral** (within-ci); mechanism: flat (1/3 predicted rows past ±2%)
- files: `muse_attention_decode/hypotheses/done/neutral/fa3-metadata-fold-preattn-full-verify.result.json`, `muse_attention_decode/hypotheses/done/neutral/fa3-metadata-fold-preattn-full-verify.patch`; branch `hyp/muse_attention_decode/fa3-metadata-fold-preattn-full-verify`
