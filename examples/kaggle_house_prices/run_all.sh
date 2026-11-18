#!/usr/bin/env bash
# End-to-end Kaggle House Prices pipeline with a local Qwen LLM.
#
#   examples/kaggle_house_prices/run_all.sh
#
# Steps: data -> prepare -> LLM server check -> baseline -> evolve (resumes if the
# run exists) -> submission from the evolved graph -> optional Kaggle submit.
#
# Tunables (env vars):
#   STEPS=40            evolution steps for this invocation (extra steps when resuming)
#   RUN_DIR=runs/house-prices
#   ISLANDS=1           number of evolution islands (each needs its own CPU slot)
#   LLM_URL=http://127.0.0.1:10000/v1
#   START_LLM=system    how to start Qwen if it is down: system (vLLM, sudo) | user (llama.cpp) | none
#   LLM_WAIT=1800       seconds to wait for the LLM server to come up (vLLM startup is slow)
#   REFINE=0            1 = PyTorch L-BFGS refinement of globals (slow on CPU slots)
#   PROMOTION=cv_sign_test  acceptance rule: cv_sign_test (significant fold sign test on CV) |
#                       cv_score (paper: any CV mean improvement) | validation_gated (CV and validation)
#   FOLDS=10            CV folds (the sign test needs >= ~9 folds to reach significance)
#   SIGN_TEST_ALPHA=0.05  cv_sign_test significance level (10 folds: 0.05 -> 9/10 wins, 0.10 -> 8/10)
#   MAX_VAL_REGRESSION=0.10  CV policies: veto candidates whose validation is >10% worse than the best
#   SKIP_BASELINE=0
#   SUBMIT=0            1 = upload submissions/evolved.csv with the kaggle CLI
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

STEPS="${STEPS:-40}"
RUN_DIR="${RUN_DIR:-runs/house-prices}"
ISLANDS="${ISLANDS:-1}"
LLM_URL="${LLM_URL:-http://127.0.0.1:10000/v1}"
START_LLM="${START_LLM:-system}"
LLM_WAIT="${LLM_WAIT:-1800}"
REFINE="${REFINE:-0}"
PROMOTION="${PROMOTION:-cv_sign_test}"
FOLDS="${FOLDS:-10}"
SIGN_TEST_ALPHA="${SIGN_TEST_ALPHA:-0.05}"
MAX_VAL_REGRESSION="${MAX_VAL_REGRESSION:-0.10}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SUBMIT="${SUBMIT:-0}"
COMPETITION="house-prices-advanced-regression-techniques"
RAW="$HERE/data/raw"
PREPARED="$HERE/data/prepared"
SUBS="$HERE/submissions"
HP="python $HERE/house_prices.py"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
[ -f .env ] || { echo "Missing $REPO/.env (local Qwen settings). See .env.example." >&2; exit 1; }

# 1. Data -------------------------------------------------------------------
log "1/6 Kaggle data"
if [ ! -f "$RAW/train.csv" ] || [ ! -f "$RAW/test.csv" ]; then
  if command -v kaggle >/dev/null 2>&1; then
    mkdir -p "$RAW"
    kaggle competitions download -c "$COMPETITION" -p "$RAW"
    unzip -o -q "$RAW/$COMPETITION.zip" -d "$RAW"
  else
    echo "Put train.csv and test.csv in $RAW (or: pip install kaggle + ~/.kaggle/kaggle.json)." >&2
    exit 1
  fi
fi
echo "using $RAW/train.csv and $RAW/test.csv"

# 2. Features ---------------------------------------------------------------
log "2/6 Prepare features"
$HP prepare --kaggle-dir "$RAW" --out "$PREPARED"

# 3. LLM server -------------------------------------------------------------
log "3/6 Local Qwen server at $LLM_URL"
llm_up() { curl -sf --max-time 5 "$LLM_URL/models" >/dev/null 2>&1; }
if ! llm_up; then
  case "$START_LLM" in
    user)   echo "starting user service: systemctl --user start qwen"; systemctl --user start qwen ;;
    system)
      # vLLM and the llama.cpp user service both bind port 10000; make sure only vLLM runs.
      systemctl --user stop qwen 2>/dev/null || true
      if ! sudo -n true 2>/dev/null && ! [ -t 0 ]; then
        echo "sudo needs a password but there is no terminal (background run)." >&2
        echo "Start vLLM first with: sudo systemctl start qwen   (or use START_LLM=user)" >&2
        exit 1
      fi
      echo "starting system service (vLLM): sudo systemctl start qwen"; sudo systemctl start qwen ;;
    none)   echo "LLM server is down and START_LLM=none." >&2; exit 1 ;;
    *)      echo "START_LLM must be user, system or none." >&2; exit 1 ;;
  esac
  waited=0
  until llm_up; do
    if [ "$waited" -ge "$LLM_WAIT" ]; then
      echo "LLM server not ready after ${LLM_WAIT}s. Check: journalctl [--user] -u qwen -e" >&2
      exit 1
    fi
    sleep 10; waited=$((waited + 10)); printf '.'
  done
  echo
fi
echo "models: $(curl -s "$LLM_URL/models" | python -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"

# 4. Baseline ---------------------------------------------------------------
if [ "$SKIP_BASELINE" != "1" ]; then
  log "4/6 Seed-graph baseline (no LLM)"
  $HP baseline --data "$PREPARED" --folds "$FOLDS" --submission "$SUBS/baseline.csv"
else
  log "4/6 Baseline skipped"
fi

# 5. Evolve -----------------------------------------------------------------
devices="$(python -c "print(','.join(f'cpu:{i}' for i in range($ISLANDS)))")"
refine_flag="--no-refine-globals"
[ "$REFINE" = "1" ] && refine_flag="--refine-globals"
if [ -f "$RUN_DIR/state.json" ]; then
  log "5/6 Resuming evolution in $RUN_DIR for $STEPS more steps"
  # Pass the task context again: without it, resume regenerates a generic one with no column map.
  evoforest-arch evolve --resume --steps "$STEPS" --env-file .env \
    --task-context-file "$PREPARED/task_context.md" --output "$RUN_DIR"
else
  log "5/6 Evolving for $STEPS steps -> $RUN_DIR"
  evoforest-arch evolve \
    --dataset external-manifest \
    --dataset-manifest "$PREPARED/dataset_manifest.json" \
    --task-context-file "$PREPARED/task_context.md" \
    --scorer rmse --folds "$FOLDS" \
    --steps "$STEPS" \
    --islands "$ISLANDS" --island-devices "$devices" \
    $refine_flag \
    --promotion-policy "$PROMOTION" \
    --sign-test-alpha "$SIGN_TEST_ALPHA" \
    --max-validation-regression "$MAX_VAL_REGRESSION" \
    --allow-source-mutations \
    --llm-provider env --env-file .env \
    --output "$RUN_DIR"
fi
evoforest-arch inspect "$RUN_DIR" | head -40 || true

# 6. Submission -------------------------------------------------------------
log "6/6 Submission from the evolved graph"
$HP predict "$RUN_DIR" --data "$PREPARED" --folds "$FOLDS" --cv --allow-source --submission "$SUBS/evolved.csv"

if [ "$SUBMIT" = "1" ]; then
  log "Submitting to Kaggle"
  kaggle competitions submit -c "$COMPETITION" -f "$SUBS/evolved.csv" -m "EvoForest + local Qwen, $STEPS steps"
fi

log "Done. Submissions: $SUBS/baseline.csv  $SUBS/evolved.csv"
