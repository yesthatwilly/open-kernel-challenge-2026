# Competition rules, local mirror

Fetched 2026-10-23 from openkernelchallenge.dev. Rules version `2026-10-22.1`
(see `rules-version.json`; `okc api` + `GET /api/v1/me` shows which version this
account accepted). The website's /rules page is client-rendered, so the text
here comes from `https://openkernelchallenge.dev/llms.txt`, the site's own
plain-text copy of the rules for agents, and the official rules PDF.

- `technical-rules.llms.txt`: the technical rules (environment, kernel and
  serving restrictions, validation, prizes, licensing).
- `official-rules.pdf` / `official-rules.txt`: Paradigm's Official Rules
  (eligibility, one account per individual, IP license, arbitration).
- `rules-version.json`: the version endpoint's response at fetch time.

Each problem's `TASK.md` carries the per-problem Rules section and `AGENTS.md`
the validation policy; those are the operational source while writing kernels.
Re-fetch if `GET /api/v1/rules` reports a newer version.
