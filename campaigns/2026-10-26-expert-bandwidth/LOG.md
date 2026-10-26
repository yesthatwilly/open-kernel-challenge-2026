# Log: campaign 2026-10-26-expert-bandwidth

The "## Now" block is rewritten by every loop tick (read it first; `head -40 LOG.md`). Entries below it are
append-only, newest last.

## Now

- updated: 2026-10-26T15:35Z
- step: X0a (judge-faithful gate inputs) + X0b (quick wins) in the worktree; then X0c ladder-2
- in flight: none
- next action: build X0a in rewrite/src/muse_decode_layer.py `_cb_data` and the moe v3 T=128/256 gate; X0b R align
  tail + combine slot prefetch; build, lint, dry run, attempt files, dispatch one decode and one moe check
- quota left 10-26 (15:00Z): decode 7 checks / 3 scored, moe 6 / 6
- boards: decode 1.4422 (rank 373 of 731), moe 1.5762 (428 of 840), overall 380 of 1222

## 2026-10-26T15:35Z: campaign opened

- Approved plan saved as PLAN.md; user decisions: single agent + one Opus audit per new CUDA module; fully
  autonomous (self-paced loop, scored submits on improvement, commit / merge / next campaign without asking).
- Predecessor 2026-10-26-decode-first-rewrite closed (RETRO.md). Code continues on branch campaign/expert-bandwidth
  forked from campaign/decode-first-rewrite ae91f55d.
- Biggest scoring gain: the NVFP4 expert block (FINDINGS.md 1-3); T=128/256 first, T <= 32 consumer second.
