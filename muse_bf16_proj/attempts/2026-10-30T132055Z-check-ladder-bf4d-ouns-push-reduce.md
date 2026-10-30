# muse_bf16_proj · 2026-10-30T132055Z · check · ladder-bf4d-ouns-push-reduce

- **Job:** `9acab0181e774ed0` (`okc status <job>`)
- **File:** muse_bf16_proj/submission.py at 6fdbd2fc (ladder/bf4d; on 63d2c5c9 = BF4c-2+3 on master aad820f8), 401,114 B


## Goal
First H200 measurement of the push cluster reduce (ouns unpooler m<=8, which routes on the judge); BF4c-2/3 ride along (fk refused on the H200 in 08569b53, so fkp likely unraced). Read the [ouns]/[fkp] lines per row.

## Result
- job `9acab0181e774ed0` (check), passed, **1.332x [1.312, 1.351]** vs board 1.321: [ounsp] push/pull m1 0.870, m8 0.932 -> ROUTE (first H200 measurement: push transfers); d1 unpooler 4.4 us (5.35), v1 unpooler 4.9 (5.57).
