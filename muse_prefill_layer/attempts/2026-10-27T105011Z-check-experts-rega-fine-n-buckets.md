# muse_prefill_layer · 2026-10-27T105011Z · check · experts-rega-fine-n-buckets

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-26T224003Z-experts-rega-fine-n-buckets.md` (branch `hyp/muse_prefill_layer/experts-rega-fine-n-buckets`).

- **Job:** `c951d9168cd546bb` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `3a8155a9`

## Goal
Built on master (blob f4751fc0). rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route block count nbk (1..8) with four buckets: nbk <= 1 -> 16, <= 2 -> 32, <= 4 -> 64, else 128. A chunk of 33-48 routes runs N = 64 and one of 65-112 runs N = 128, so up to 3/8 of those items' MMA columns and slab rows are padding. Judge-like skewed routing has many such chunks: turn_1 experts of ~33-112 routes, and the tail chunk of every k >= 2 expert at turn_4 and first_turn (uniform in 1-128 routes). Prepare's KERNELS lines put a stage at ~0.60 us at N = 32 (pingpong log, T=753 random: 307 us for 16 items x 32 stages) and ~0.93 us at N = 128 (T=3012 random: 474 us), about 3.4 ns per N row per stage, so N = 16 * nbk saves up to ~0.16 us per stage on those items. CHANGE: derive a module muse_prefill_rega_fine from master's source (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): (1) generate Wg<NB> for NB in (16, 32, 48, 64, 80, 96, 112, 128) (_rega_wgmma_src already emits the m64nNk16 RS wrapper per NB; bf16 wgmma takes any N that is a multiple of 8 up to 256); (2) struct Maps holds 8 CUtensorMaps and rega_gemm encodes 8 boxes {64, 16*(b+1)} with master's SWIZZLE_128B 2-D map (every box fits the 16 KB slab half); (3) decode_item sets m.bucket = nbk - 1 (clamped to 0..7); (4) the producer's tx = WST + 2 * 16 * (m.bucket + 1) * 128; (5) the consumer switch has 8 cases calling consume_item<16*(b+1), G>. NB/8 is even for every NB, so the epilogue loops are unchanged; the dequant, ring, smem layout and register budget (largest N still 128) are master's. GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y with master's rega on random, skewed, one and pop routings (each output element is the same K-ordered dot product whatever N), plus the two-state replay check; then 2 interleaved rounds x 11 cold-L2 graph alternations, master pair vs fine pair, on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and random <= 1.00; state['rega_fine_ok'][cls]. Print per class and routing the chunk histogram by nbk, KERNELS w13 / w2 us both ways, PATH CHOSEN. The build has 32 consume_item instantiations instead of 16: if prepare passes ~230 s, skip _rega_lo_phase (its candidates have been OFF in every run since combo-b).

## What changed, and where
Worktree `.worktrees/muse_prefill_layer/experts-rega-fine-n-buckets`, `muse_prefill_layer/submission.py` only (base master ec785165, blob f4751fc0; uncommitted). Diff: 1 file, +618 / -4. Idiom copied from the sibling `hyp/muse_prefill_layer/experts-rega-round-rotated-schedule` (same base blob): separate module, asserted string edits of master's source, master's kernel / v1 fallback / sha256 checks untouched, phase last in prepare().

What changed, where:
- `_rega_wgmma_src(buckets=REGA_BUCKETS)`: a default parameter only. Master's call sites pass nothing, so master's text is byte-identical (checked offline: `sha256(_REGA_CPP + _rega_src())` still equals `REGA_V2_SHA256`, and `_rega_src_v1()` still derives). The fine module passes `REGA_FINE_NB = (16, 32, 48, 64, 80, 96, 112, 128)`.
- New module `muse_prefill_rega_fine` (`_rega_fine_src`): master's head + 8 wgmma wrappers + master's body through `_REGA_FINE_EDITS`. Each edit has a required occurrence count, and any other count means no build. After the edits there are sanity checks: no `rega_kernel<`/`rega_kernel(`/`16 << `/`rega_gather_kernel` left, both decode_item calls pass `fine`, 9 `WG_CONSUME(`, `CUtensorMap m[8]`. The edits:
  - `Maps` grows to 8 CUtensorMaps, with `static_assert(sizeof(CUtensorMap) == 128 && sizeof(Maps) == 8 * 128)` (a 1 KB `__grid_constant__` parameter). The tensor maps are still encoded host-side in `rega_gemm_fine` from that call's bsrc, as master's rega_gemm does, so this happens at capture and never inside a replay. The 8 boxes are `{64, 16 * (b + 1)}` with master's SWIZZLE_128B 2-D map.
  - `decode_item(..., int fine)`: `fine = 1` gives `bucket = clamp(nbk - 1, 0, 7)`; `fine = 0` is master's mapping `nbk<=1?0:nbk<=2?1:nbk<=4?3:7`, the overhead control. The producer thread and the consumer warpgroups both call decode_item with the kernel's `fine` argument, so they pick the same bucket, map, tx and consume_item<NB>. Master's bounded mbarrier wait (error word 1) is kept, so a mismatch would turn a class OFF rather than hang.
  - Producer `tx = WST + 2 * (16 * (m.bucket + 1)) * 128`.
  - The consumer switch has 8 cases, `consume_item<16 * (b + 1), G>`. That is 32 consume_item instantiations, against master's 16.
  - Renames: `rega_kernel_fine`, `rega_init_fine`, `rega_gemm_fine(..., int64_t fine)` (with a TORCH_CHECK that fine is 0 or 1), `rega_gather_fine` and `rega_gather_fine_kernel` (unused, renamed as the rot sibling did). Dequant, ring, smem layout, epilogue and the 232-register consumer budget are master's.
- `_experts_rega`: `fine = not variant and rega_fine_ext is not None and rega_fine_ok[_rega_class(T)]`. When it is set, both GEMMs call `rega_gemm_fine(..., REGA_FINE_MODE=1)`; otherwise the dispatch is master's, unchanged. The decision uses shape and prepare-time values only. The `rega_gemm_lo` path (variant != 0) keeps precedence.
- `prepare()`: `_rega_fine_phase` runs after `_rega_lo_phase`, which itself runs after `_rega_setup`, `_fa3c_setup` and `_norm_gather_setup`, so it is last of every phase that has its own deadline. Its summary lines are printed last, truncated to 700 chars each, after the reprinted align2 / fa3c / lo lines, and its PATH CHOSEN lines are the final lines of prepare.
- `_rega_fine_phase` / `_rega_fine_class` / `_rega_fine_exact` / `_rega_fine_time`, per the review's binding items:
  1. **Low-class precedence:** the low class is gated only when `rega_low_variant == 0`, against master's rega_gemm, which is the incumbent actually chosen then. Otherwise it is OFF with the reason. `rega_low_variant` is printed in the low GATE line and in PATH CHOSEN low.
  2. **Budget:** the phase is skipped with an OFF line, before the source derivation and the nvcc build, when prepare is past `REGA_FINE_START_DEADLINE_S = 200 s`. Per-class start deadlines: low and high 248 s, ft 236 s. Per-routing timing deadlines: 258 / 258 / 248 s. Replay-check deadlines: 272 / 270 / 260 s. The phase line logs the build seconds, the seconds per class, the phase total and the prepare total.
  3. **Gate:** per class, fine pair / master pair <= 0.98 on pop in both rounds AND <= 1.01 on random with the rounds pooled. The timing pass is followed by the two-state A/B/C replay check, with fine ON for that class (`REGA_FINE_STATE_KEYS`).
  4. **Overhead control:** the timing runs three pairs (m = master, f4 = fine module with the 4-bucket map, f = fine), plus each kernel alone, over 2 rounds x 11 interleaved cold-L2 alternations, with an empty-graph baseline subtracted. A KERNELS line per routing prints m/f4/f us for the pair (per round and pooled), f4/m, f/f4 and f/m, and w13 and w2 pooled.
  5. **Exactness:** hs and y must be bit-identical to master's rega_gemm on random, skewed, one and pop, with each w2 fed master's hs. On a miss, max |diff| (hs and y) and the first mismatching (class, routing, nbk) are printed. The fallback then holds the whole `_experts_rega` block, with fine ON, to master's fp32-reference calibrated bound: max(2 x the incumbent's error, 8e-3), worst row 5x, error word 0 and both markers. If that also fails, the class stays OFF. The f4 == master check is print-only ("control suspect" if it fails).
  6. **nbk histogram:** chunks by nbk 1..8 on the pop routing per class, with the local routes and the mean / max routes per touched expert, to compare with the judge's ~42 routes per expert at turn_1. The live log has the histogram for every exact routing.
  7. **Maps:** as above: 8 maps, static_assert'ed, host-side encode.
  8. **Build:** by asserted string edits, under a separate module name. Master's kernel, the sha256 checks and the v1 fallback are untouched. `_rega_lo_identity()[0]` (master's v2 hash) must hold before the build.
  9. **Producer / consumer:** the same `fine` reaches both, through the same decode_item. Master's error-word and bounded-wait path is kept.
  10. **Log tail:** the end block is compact (per class: nbk, exact, KERNELS pop, KERNELS random, GATE), then the phase seconds, then the PATH CHOSEN lines last.

Deviations from the md's `change` paragraph, and why:
- The md said "if prepare passes ~230 s, skip _rega_lo_phase". Not done: the review's binding item 1 / 2 puts the fine phase after `_rega_lo_phase` with its own skip at 200 s, so master's lo phase is untouched.
- The random threshold is 1.01, not the md's 1.00 (review item 3). The first_turn expectation is revised to flat (review item 3). Both the frontmatter `expected` and the Expected bullet were edited.
- The fine-4map control is the same compiled kernel with a runtime int, not a separate instantiation. This keeps the build at 32 consume_item instantiations, and it makes f vs f4 a pure bucket-mapping effect, because the binary is identical.

Uncertainties (nothing could be run on this node: no GPU, no nvcc):
- The n48 / n80 / n96 / n112 wgmma forms and the 48 / 80 / 96 / 112-row SWIZZLE_128B TMA boxes have not been compiled or run in this repo. A build failure, canary failure or exact-and-bound failure leaves master's rega_gemm.
- Build time: 32 consume_item instantiations, against master's 16 (master's builds took 31-41 s). With prepare at about 143 s before the phase (d00770b6), the build should finish around 190-205 s. ft (deadline 236 s) may be skipped by time, which is acceptable because ft is expected flat.
- Bit-exactness across N assumes wgmma accumulates each output column in the same k order whatever N is. If it does not, the fp32-bound fallback decides, and the md reports `exact ... NO, fp32 bound ok`.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-fine-n-buckets diff`:

- [x] CUDA-graph capturable: the only kernel_fn-reachable change is in `_experts_rega`: a dict lookup of prepare-time bools, `_rega_class(T)` (shape-only), and an extension call with the same argument pattern as master's rega_gemm. There is no `.item()`, `.cpu()`, sync or print, and no allocation or control flow on input values. Tensor maps are encoded host-side per call from shapes and pointers at capture, as master does.
- [x] Nothing derived from `data` is cached across calls: state gains `rega_fine_ok` (plain bools from prepare's gate) and `rega_fine_ext` (the module). No module globals hold tensors (master's `_rega_audit_globals` scan still applies, and the new globals are ints, strings and tuples).
- [x] Weights: no new weight storage. The fine kernels read master's `rega_bank` (lossless NVFP4 code/scale retiling and fp32 global scales, made in prepare). NVFP4 stays NVFP4; dequant stays in registers (master's dq_stage).
- [x] Activations BF16 where SGLang keeps BF16: xs, hs and y are bf16, with fp32 accumulation and master's epilogue (silu*up in fp32, one bf16 rounding). KV cache and queries are untouched (attention is not changed).
- [x] n/a: attention is not touched; every key is still attended (no change outside the expert GEMMs).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only new build is a `load_inline` CUDA extension, as master builds its own; lint reports no process/network/harness findings.
- [x] All work happens inside the timed call; no output caching or replay: each kernel_fn call runs align, gather, both GEMMs and the weighted sum, exactly as master does.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`): exit 0, "OK with warnings". There are 105 warnings, against master's 93. The 12 new ones are prepare-only `host-sync` warnings in `_rega_fine_*` (outside kernel_fn's call graph). No errors. `python3 -I -m py_compile` passed, and its `__pycache__` was removed.

```
warn  host-sync    L469: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L511: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L545: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L601: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L607: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L617: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L622: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L626: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L632: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L637: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L642: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L805: sorted_ids = sorted_ids.cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L806: block_experts = block_experts.cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L807: used = min(int(n_padded.cpu()[0]) // bsm, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L810: blocks = (block_experts[:used] == e).nonzero().flatten().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L814: routes += seg[seg < n].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L829: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L852: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L902: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1095: idx = (a != b).nonzero().flatten()[0].item()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1096: return f"{name}[{idx}] master {a.view(-1)[idx].item()} vs new {b.view(-1)[idx].item()}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1114: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1153: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1173: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1185: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1311: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1316: err = ((g - r).abs() / (r.abs() + 1e-3)).max().item()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2058: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2065: view = tensor[:int(rows.item())]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2136: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2140: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2180: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2187: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2188: word = int(states[i]["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2189: marks = states[i]["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2204: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2224: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2298: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2299: word = int(state["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2300: marks = state["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2326: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2328: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2329: word = int(state["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2330: marks = state["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2360: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2361: prof = state["ssdq_prof"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3461: q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L3462: q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L3624: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3627: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3665: word, m13, m2 = [int(v) for v in err.tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3720: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3721: be = block_experts[:used].cpu().long()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3746: pads.append(torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device=device))  (allowed only where SGLang does the same)
warn  host-sync    L3751: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3783: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3788: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3807: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3846: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3904: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3906: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3935: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3941: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4051: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4061: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4081: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4083: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4147: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L4293: return (torch.empty(shape, dtype=torch.uint8, device=device), torch.empty(shape, dtype=torch.uint8, device=device))  (allowed only where SGLang does the same)
warn  host-sync    L4307: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4326: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4345: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4399: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4442: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4471: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4483: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4598: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4617: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4656: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4659: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4680: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4819: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4826: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5409: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5410: be = block_experts[:used].cpu().long()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5413: routes = torch.bincount(lids[lids >= 0].long().cpu(), minlength=LOCAL_EXPERTS)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5463: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5468: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5537: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5556: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5574: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5628: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5701: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6017: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6018: be = block_experts[:used].cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6041: routes = torch.bincount(lids[lids >= 0].long(), minlength=LOCAL_EXPERTS)[:LOCAL_EXPERTS].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6084: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6108: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6136: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6159: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6179: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6182: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6216: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6360: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L6417: k_buf, v_buf = (torch.empty((batch, width, 2, 64), dtype=torch.uint8, device=q8.device)  (allowed only where SGLang does the same)
OK with warnings: muse_prefill_layer/submission.py
exit 0
```

## Risks / what could fail
medium
The n48 / n80 / n96 / n112 wgmma shapes are standard sm_90a forms but have not run on the judge in this repo; the separate module means a build or canary failure keeps master's rega. Size ~+1-1.7% headline if the judge-like bucket mix holds; read the nbk histogram and the KERNELS lines.

## Result (after the run)
- job `c951d9168cd546bb` (check), status done, passed True, tests 5/5
- speedup 1.4573x, 95% CI [1.4355, 1.4792], score 265828.0282765953 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4677493909229, "ci": [1.4457331500590564, 1.4897656317867434], "job": "73b40cd13fdf4bb0"}; base result: muse_prefill_layer/hypotheses/done/neutral/combo-b-normgather-two-cta-low-regate.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-fine-n-buckets.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-fine-n-buckets.patch`; branch `hyp/muse_prefill_layer/experts-rega-fine-n-buckets`
