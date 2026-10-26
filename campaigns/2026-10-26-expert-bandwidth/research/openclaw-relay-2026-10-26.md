# Relay notes from the user's openclaw session, 2026-10-26

Written by the Claude session on `openclaw` that read this repo over claw (read-only, except the steer
banners) and relayed the user's steers. These findings exist only in that session. Fold what is useful into
DISTILL.md / FINDINGS.md.

## 1. Where decode's time goes, judge-weighted (check f59cb301, 1.5347)

Shares of (46·sliding + 16·full) time per load. The geomean over the 6 loads makes the headline row the average,
which is also the headline's sensitivity: cutting an area's time by x% raises the headline by about share·x%.

| load | moe block (experts + route/align/combine) | bf16 dense GEMMs | attention | small ops |
|---|---|---|---|---|
| decode 1 | 35% | 44% | 14% | 7% |
| decode 16 | 53% | 27% | 17% | 4% |
| decode 32 | 53% | 22% | 23% | 3% |
| verify 1 | 59% | 29% | 8% | 4% |
| verify 16 | 62% | 18% | 12% | 1% |
| verify 32 | 60% | 18% | 14% | 2% |
| **headline** | **53%** | **26%** | **15%** | **3%** |

- Classification by kernel name. moe: ea_w*/ea4_*/wg_kernel/combine/route_norm_align/_hc_align/post_expert.
  dense: pooler_router/swapab/nvjet/dl_qkv/dl_o/dl_unpool/cublasLt. attention: sliding_fused/splitkv/flash*/skw_*/
  _pre_attention. small: norm_residual/at::native.
- About 2% sits in "... N more" rows that the profiler did not list.
- Optimistic ceilings from 1.5347, using `1/(1 - share·cut)`:
  - moe, cut to x0.6-0.7 (ring ceiling ~4.4 TB/s vs ~3.0 real): x1.19-1.27, so 1.83-1.95.
  - bf16 dense, cut 10-15% (incumbents at 3.4-3.9 TB/s; bulk12 breaks the raw-byte floor, so more is possible at
    T=1): x1.03-1.04.
  - attention: the plan's ≤x1.061, realistically x1.014-1.032.
  - small ops halved: x1.017.
  - Everything outside moe stacked: about x1.09. Everything stacked: about x1.30, so ~2.0 (a ceiling, not a forecast).

## 2. Steering an autonomous loop from outside: what reached it and what did not

- **Did not reach it:** a bullet inserted into the LOG.md "## Now" block. The loop writes Now from a scratchpad template
  (`now.md`, `log_now.py`) over the file without reading it first. It overwrote a steer twice unread: at 18:49Z, and at
  19:05:50Z seconds after the second one landed.
- **Did not reach it, at least not by itself:** a bullet in PLAN §6 or §11. The tick prompt names §6, but a long-context
  agent did not re-read it.
- **Worked within minutes:** a blockquote banner at the top of `campaigns/INDEX.md` and after PLAN.md's frontmatter, plus
  a campaign-folder `CLAUDE.md`. The agent noticed "CLAUDE.md, PLAN.md and INDEX.md changed again on master", read them,
  acknowledged them in LOG and acted, for all three steers (19:05Z quota policy, 21:40Z sprint, ~22:00Z sprint stop).
- Lesson for the loop: read the current Now block before rewriting it, and keep any bullet marked as the user's.

## 3. Operating-cost observations

- Main session from 21:00Z to 21:46Z, subagents excluded: 83.3M cache-read, 0.40M cache-write, 0.27M output tokens. The
  last call carried about 627k tokens of context. Nearly all calls were execution (grep / sed ranges / edits / builds /
  status), each re-reading the whole context.
- Idle heartbeats of 3600 s risk prompt-cache expiry, which re-writes the full context at the cache-write price.
- No usage-limit events in the transcript; the 27 "rate limit" hits are instruction text.
- The user's direction (22:1xZ): use the heavy context for distillation and planning (cache hits that produce durable
  plans), and run execution in a fresh, light session with subagents.

## 4. Other notes

- bulk12 rules reasoning (attempt 2026-10-26T2029Z): the codec is lossless for any weights, with escapes for
  out-of-window values (judge rate 1.1-1.4e-4, ≤58/CTA, cap 224); only its speed depends on the distribution. The
  reasoning is sound, and worth stating in any final rules audit.
- Earlier LOG Now stamps (19:30Z / 19:50Z) were estimates, later corrected. Use `date -u` for every stamp.
- Quota on 10-26 after the sprint stopped: about bf16 7 checks / 9 scored, attention 14 / 10, prefill 15 / 10, all unused
  at the reset.
