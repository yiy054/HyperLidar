#!/bin/bash

# -------------------- CONFIG --------------------
DATASET_PATH="/mnt/data/semantickitti"
ARCH_CONFIG="config/arch/senet-512.yml"
MODEL_NAME="senet-512"
VAL_SEQ=8
ALL_SEQS=(0 1 2 3 4 5 6 7 9 10)
# -------------------- PREPARATION --------------------

mkdir -p pipeline_logs

# Function to join array by commas
join_by_comma() {
  local IFS=","
  echo "$*"
}

# -------------------- MAIN LOOP --------------------
# Generate all combinations of 8 out of the 10 sequences
COMBS=$(python3 -c "
from itertools import combinations
print('\n'.join(','.join(map(str, c)) for c in combinations([0,1,2,3,4,5,6,7,9,10], 8)))
")

# List of comma-separated combinations to skip
SKIP_LIST=("0,1,2,3,4,5,6,7" 
        "0,1,2,3,4,5,6,9" 
        "0,1,2,3,4,5,7,9" 
        "0,1,2,3,4,6,7,9" 
        "0,1,2,3,4,5,6,10" 
        "0,1,2,3,4,5,7,10"
        "0,1,2,3,4,5,9,10"
        "0,1,2,3,4,6,7,10" 
        "0,1,2,3,4,6,9,10"
        "0,1,2,3,4,7,9,10")

# Filter out skipped combinations
FILTERED_COMBS=$(echo "$COMBS" | while read line; do
  skip=false
  for skip_entry in "${SKIP_LIST[@]}"; do
    if [[ "$line" == "$skip_entry" ]]; then
      skip=true
      break
    fi
  done
  if ! $skip; then
    echo "$line"
  fi
done)

for PRETRAIN_STR in $FILTERED_COMBS; do
    # Convert to array
    IFS=',' read -r -a PRETRAIN <<< "$PRETRAIN_STR"

    # Get RETRAIN = ALL_SEQS - PRETRAIN
    RETRAIN=()
    for seq in "${ALL_SEQS[@]}"; do
        skip=false
        for p in "${PRETRAIN[@]}"; do
            if [[ "$seq" == "$p" ]]; then
                skip=true
                break
            fi
        done
        if ! $skip; then
            RETRAIN+=("$seq")
        fi
    done

    PRETRAIN_ID=$(echo "${PRETRAIN[*]}" | tr -d ' ,')
    RETRAIN_STR=$(join_by_comma "${RETRAIN[@]}")
    PRETRAIN_STR_COMMA=$(join_by_comma "${PRETRAIN[@]}")
    LOG_TAG="pretrain${PRETRAIN_ID}"
    NOW=$(date "+%Y%m%d_%H%M%S")
    LOG_FILE="pipeline_logs/pipeline_${LOG_TAG}_${NOW}.log"
    MODEL_PATH="${LOG_TAG}/${MODEL_NAME}"

    echo ""
    echo "=================== $LOG_TAG ==================="
    echo "Pretrain: ${PRETRAIN_STR_COMMA}"
    echo "Retrain:  ${RETRAIN_STR}"
    echo "Log:      ${LOG_FILE}"
    echo "================================================"

    # Pretrain
    python3 train.py \
        -d "$DATASET_PATH" \
        -ac "$ARCH_CONFIG" \
        -n "$MODEL_NAME" \
        -l "$LOG_TAG" \
        -t "$PRETRAIN_STR_COMMA"

    # Retrain/Infer
    python3 main.py \
        -d "$DATASET_PATH" \
        -l "infer_$LOG_TAG" \
        -m "$MODEL_PATH" \
        -s valid \
        -t "$RETRAIN_STR" \
        > "$LOG_FILE" 2>&1
done

echo "✅ All combinations completed."