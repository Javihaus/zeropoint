# Results — Tetris decision bench (CPU)

**Date:** 2026-11-27  
**Runtime:** Google Colab, CPU  
**Notebook:** `../tetris_decision_bench.ipynb`

## Models measured in this experiment

- `greedy` — shortlist ceiling  
- `random` — shortlist floor  
- `zeropoint` — ZeroPoint 1.1.0, local CPU  
- `laya` — Laya zero-shot, CPU-forced  

## Not in this experiment

**OpenAI Decisions was not run.**  
The notebook can call it later (`run_openai = True` + API key), but these CSVs and plots do **not** include OpenAI results.

## Protocol

- seeds: 1–10  
- max_moves: 100  
- shortlist_k: 8  
- board: 10×20, 7-bag, drop placements only  
- Laya: `CUDA_VISIBLE_DEVICES=""`, zero-shot (no Tetris fine-tune)

## Files

| File | Contents |
|------|----------|
| `summary.csv` | Aggregated metrics |
| `games.csv` | Per-seed rows |
| `smoke_games.csv` | Smoke test |
| `summary_plots.png` | Lines / latency / survival |
