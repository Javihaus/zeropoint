# STATUS

Updated 2026-11-05T23:09:16. Last event: finished report (exit 0)

| stage | done |
|---|---|
| PROTOCOL.md written (frozen) | yes |
| 1 lists (cache/lists_*.json, results/lists_info.json) | yes |
| 0 packaging (results/packaging.json) | yes |
| 2 cache E1 (17/17 splits started; train parts 30/30) | yes |
| 2 cache E2 (17/17 splits started; train parts 30/30) | yes |
| 3 depth analysis (results/depths.json) | yes |
| 4 train E1 (25/25 runs) | yes |
| 4 train E2 (25/25 runs) | yes |
| 5 eval + bootstrap (results/metrics.json, results/paired.json) | yes |
| 6 report (results/REPORT.md, results/summary.json) | yes |

## How to continue

Run `bash scripts/driver.sh` from the ablation folder (in the background). Every stage skips work whose
output file exists, so a rerun resumes. Logs are in `logs/`. Python is `pkg/venv/bin/python -I`.
Deviations from PROTOCOL.md, if any, are listed below and in REPORT.md.

## Deviations

- Operational, not a change of protocol: the container's disk filled up (about 43 GB usable, mostly taken by earlier sessions' files) while caching E2. After all 25 E1 runs were saved, the E1 feature arrays (X, cls) were dropped from cache/E1 by scripts/slim_cache.py, keeping offsets, gold indices and released depths. The E2 caching was relaunched and resumed from its saved parts. Retraining E1 would require caching its features again.
- Operational: torch was set to one thread per training process (small operations are about 35 times faster than with two threads on this CPU); this changes speed, not the computation.
