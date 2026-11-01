# Retired 2026-11-01: pre-public-foundation queue (12 files)

Every file here was minted on a master `submission.py` blob from before the 10-31 ~05:10Z user ruling that rebased
master onto the public alygg77@6cae49a kernels (campaigns/INDEX.md; our old files are at tag `ours-final-1031`).
None matches master's current blob, so churn would only ever hold them as stale-blob, and reviewing them spends
tokens on plans for code that no longer exists. Moved here (not deleted) on the user's request (11-01: "i hate the
idea of wasting tokens going over and over reviewing plans that are obsolete").

- churn (`all_hyps`: queue, running, done/*) and `/hypothesize` (`hyp.py context`) do not read this folder.
- The mechanism-level lessons stay in `../../CLAIMS.md`, which still guards re-minting refuted mechanisms.
- To revive one: re-specify it against master's current code with `hyp.py new --prior "<old slug>: <what differs>"`;
  do not `git mv` it back unchanged (its line numbers and kernels refer to the old files).
