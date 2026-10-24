# muse_attention_decode · 2026-10-24T120945Z · check · preattention-group-tiled-programs

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T120513Z-preattention-group-tiled-programs.md` (branch `hyp/muse_attention_decode/preattention-group-tiled-programs`).

- **Job:** `877731a3b2214ddb` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `9c4ab08`

## Goal
Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (token, kv group): grid (tokens, Hkv), num_warps=4. Program (t, g) loads the G = Hq/Hkv = 8 query heads of group g as one [G, D] tile (rows h = g*G + r, columns even/odd halves as now, or a [G, D] tile plus a pair-swapped copy for RoPE as _norm_rope in the sliding-single-launch patch does), plus key head g and value head g as [1, D] rows; it applies the same scaleless RMSNorm per row (sum over axis 1, eps 1e-5, fp32), the same interleaved RoPE from positions[t] and the cached inv_freq when ROPE, writes the 8 FP8 query rows (in the PAIRED layout, query h writes its half (h // G) % 2 of its 128-byte padded row and FP8 zeros into the other half, every byte written each call, exactly as now), and stores the FP8 key and value rows divided by k_scale/v_scale at cache_loc[t]. Arithmetic, rounding points and output layouts are identical to master; only the decomposition into programs changes. kernel_fn, FA3 and both epilogues are unchanged. Mechanism: the kernel moves under 1.2 MB even at verify_32 (0.25 us at 4.8 TB/s) yet takes 4.2 us there, 2.7-2.8 us at verify_16 and ~2 us at decode shapes: it is bound by issuing thousands of single-warp CTAs each doing a dependent load-reduce-store chain, so 9x fewer, 4x wider programs should cut it toward ~1-1.5 us.

## What changed, and where
- Only `_pre_attention_kernel` and its launch in `kernel_fn` changed (worktree `muse_attention_decode/submission.py`; diff 62+/48-, most of it the kernel body and docstrings).
- Grid `(tokens, hq + hkv)` with `num_warps=1` becomes `(tokens, hkv)` with `num_warps=4`. Program `(t, g)` loads the group's G query heads (`heads = g*G + arange(G)`) as two `[G, HALF]` even/odd tiles, and key head g as two `[HALF]` vectors (the key is kept 1-D as in master, not as a `[1, D]` tile). Value head g is loaded as `[D]`.
- The math is the same as master, per row. RMSNorm is `sum(x_e^2, axis=1) + sum(x_o^2, axis=1)`, then `1/sqrt(ss/D + 1e-5)` in fp32. RoPE reads `positions[t]` and the cached `inv_freq` and computes `cos`/`sin` once on `[HALF]`, broadcast over the G query rows and reused for the key. The cast points are unchanged: FP8 for q, `k/k_scale` and `v/v_scale` to FP8.
- PAIRED query store: `half = g % 2`, which equals master's `(h // G) % 2` because `h // G == g` for every query in the group. Each of the G padded rows gets its D values in `half*D` and a `[G, D]` FP8 zero tile in `(1-half)*D`. That covers all `G * 2D` bytes of the group's rows, so with `hkv` groups every byte of `[tokens, HQ, 2D]` is written on every call. The unpaired path writes `(t*HQ + h)*D`, which covers `[tokens, HQ, D]` fully. Both constexpr branches were regridded, as the review asked.
- K/V stores go to `(loc*HKV + g)*D` with `loc = cache_loc[t]`, the same address as master's `kh = h - HQ`, so the page-64 layout is unchanged. Each of the `tokens*hkv` programs writes exactly one k row and one v row.
- Deviation: none in behavior. Docstrings were updated, and one constraint is now stated: G must be a power of two (`tl.arange`). G = 8 under both tp8 (16/2) and dp (128/16), so every test and scored shape satisfies it.
- No shape gate was added. Per the review, a batch-1 regression is a follow-up decision based on this run's per-shape `_pre_attention_kernel` rows. num_warps=2 is also a later ab-tune.
- Not run: this node has no GPU, Triton or torch. Correctness and compile were not exercised beyond `py_compile`.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/preattention-group-tiled-programs diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The only host-side change is the grid tuple and num_warps, both derived from config (`tokens`, `hkv`). There is no new allocation and no branching on input values.
- [x] Nothing derived from `data` is cached across calls. `_INV_FREQ` is unchanged and keyed on config theta and d only.
- [x] n/a: this problem has no weights or `prepare()`. The diff touches no weight path.
- [x] Activations: the qkv reads are bf16 to fp32, the KV cache is written as FP8 E4M3 (`float8e4nv`), and queries are FP8 E4M3, the same as master. The diff changes no precision.
- [x] Attention sees every key: FA3 call, window and page table are untouched by the diff.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none of these appear in the diff (grep clean).
- [x] All work happens inside the timed call, and nothing is cached or replayed: the kernel still runs in full inside `kernel_fn` on every call.
- [x] First line `# okc: problem=muse_attention_decode` kept. No `triton.autotune`.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/preattention-group-tiled-programs/muse_attention_decode/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left was removed, and none remains in the worktree.

## Risks / what could fail
low
Low correctness risk (same math, same stores); the reviewer should check that the paired zero-half store and the value store remain complete for every byte and that the key program per group still writes to cache_loc with the page_size-64 test layout ((loc * HKV + kh) * D is unchanged). Perf risk at batch-1 shapes: 2 programs of 4 warps per token instead of 18 of 1 may lengthen the critical path by a few hundred ns; if so, a companion with grid (tokens, Hkv, 2) (4 query heads per program) is the fallback. Independent of fa3-scheduler-metadata-side-stream except that a shorter pre-attention kernel leaves less time to hide the side-stream prepare kernel under; they should be measured separately.

## Result (after the run)
- job `877731a3b2214ddb` (check), status done, passed True, tests 16/16
- speedup 1.2614x, 95% CI [1.2425, 1.2803], score 2985.8227792583984 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2686474414890054, "ci": [1.2496177298666704, 1.2876771531113405], "job": "bd64bf9f9e904f02"}; base result: muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.result.json`, `muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch`; branch `hyp/muse_attention_decode/preattention-group-tiled-programs`
