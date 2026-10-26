# Research for campaign 2026-10-26-decode-first-rewrite

Gathered during planning on 2026-10-25/26 (UTC). The three reports are copied verbatim from read-only
Explore subagents (Opus); do not edit them. Notes and corrections go in `../FINDINGS.md`.

| file | what it is |
|---|---|
| `decode_layer_read.md` | muse_decode_layer: board vs master, per-shape time budget, ceilings, proven/dead/stranded wins, path map per T band, port-lag table, ROI paths, queue verdicts |
| `nvfp4_engine_forensics.md` | NVFP4 expert GEMMs: shared dequant cost, per-kernel structure and limiter tables, measurement confounders, instruction budget, exact cheaper dequant with proof, Engine A and B designs, the R0–R6 microbenchmark ladder |
| `dense_gemm_forensics.md` | dense bf16 GEMMs: per-cell floors and gaps, why m≥128 custom kernels lost, the one-M-tile x-multicast split-K design, m≤32 options, the port map to decode |
| `board_fit.py`, `board_fit.out.txt` | fits the overall-board ordering against the four kernel boards and prints the decode speedup needed per overall rank |
| `roi_model.py`, `roi_model.out.txt` | splits decode profile a903b9d0 into component classes and recomputes the headline under component scenarios |
| `leaderboard-2026-10-26/` | `okc leaderboard --json` snapshots (overall and each problem), gzipped |

Re-run:

```
python3 -I board_fit.py            # reads leaderboard-2026-10-26/
python3 -I roi_model.py [profile.json.gz]
```
