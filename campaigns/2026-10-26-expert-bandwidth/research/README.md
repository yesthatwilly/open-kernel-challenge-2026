# Research for campaign 2026-10-26-expert-bandwidth

Gathered on 2026-10-26 (UTC) during planning. The two reports are the verbatim structured outputs of read-only agents
(workflow run wf_7532bad1-294); do not edit them. Notes and corrections go in `../FINDINGS.md`.

| file | what it is |
|---|---|
| `expert_engine_forensics.md` | v3 / Engine A / Engine B limiters with code and log citations; designs S1, D1, D2, D3; ladder-2 rungs and verdict rule; rules notes |
| `secondary_levers.md` | attention (board vs decode kernels, the reopened full-layer verdict), dense m=128/256, T<=32 glue; ranked by the judge formula |
| `board_structure.py`, `board_structure.out.txt` | overall-board structure (decode order + composite for non-decode teams) and the decode speedup -> rank map |
| `lever_model.py`, `lever_model.out.txt` | headline formula check and component scenarios on 0d84f58ca33b424b |
| `leaderboard-2026-10-26T15Z/` | `okc leaderboard --json` snapshots (overall and all five problems), gzipped |

Re-run:

```
python3 -I board_structure.py
python3 -I lever_model.py ../../2026-10-26-decode-first-rewrite/runs/0d84f58ca33b424b.profile.json.gz
```
