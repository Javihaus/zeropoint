#!/usr/bin/env bash
# Resumable pipeline. Usage: bash scripts/driver.sh [stage ...]   (default: every stage, in order)
# Stages: lists packaging cache_E1 cache_E2 depths train_E1 train_E2 eval report
# Each stage skips work whose output already exists, so rerunning resumes where it stopped.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/pkg/venv/bin/python -I"
S="$ROOT/scripts"
mkdir -p logs results cache
SPLITS="dev_stop dev_cal bench2000 snips atis mtop_es hwu64 clinc_real_indomain tickets_made tickets_far clinc_20 clinc_50 clinc_100 clinc_150 mtop_es_20 mtop_es_50 train"
STAGES="${*:-lists packaging cache_E1 cache_E2 depths train_E1 train_E2 eval report}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

st() { $PY "$S/status.py" "$1"; }
for stage in $STAGES; do
  st "starting $stage"
  case $stage in
    lists)     N_SUB=30000 $PY "$S/s1_lists.py" >> logs/lists.log 2>&1 ;;
    packaging) $PY "$S/s0_packaging.py" >> logs/packaging.log 2>&1 ;;
    cache_E1|cache_E2)
      enc=${stage#cache_}
      if [ ! -f "cache/$enc/DONE" ]; then
        $PY "$S/s2_cache.py" "$enc" $SPLITS >> "logs/$stage.log" 2>&1 && touch "cache/$enc/DONE"
      fi ;;
    depths)    $PY "$S/s3_depths.py" >> logs/depths.log 2>&1 ;;
    train_E1|train_E2) $PY "$S/s4_train.py" "${stage#train_}" >> "logs/$stage.log" 2>&1 ;;
    eval)      $PY "$S/s5_eval.py" >> logs/eval.log 2>&1 ;;
    report)    $PY "$S/s6_report.py" >> logs/report.log 2>&1 ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
  rc=$?
  st "finished $stage (exit $rc)"
  echo "stage $stage exit $rc"
  if [ $rc -ne 0 ]; then exit $rc; fi
done
