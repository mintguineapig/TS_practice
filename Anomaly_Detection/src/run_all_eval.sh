#!/bin/bash
# Evaluate LSTM_AE and USAD on all 5 datasets
# Results are logged to eval_results.log

export CUDA_VISIBLE_DEVICES=0
cd /workspace/dsba/Time-series/Time-series-AD_2025/src

LOG="eval_results.log"
echo "========================================" | tee -a $LOG
echo "Evaluation started: $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

DATASETS=("PSM" "SMD" "MSL" "SMAP" "SWaT")

# ── LSTM_AE ──────────────────────────────────────────────────────────
for DS in "${DATASETS[@]}"; do
    echo "" | tee -a $LOG
    echo ">>> LSTM_AE | Dataset: $DS | $(date)" | tee -a $LOG
    echo "---" | tee -a $LOG
    python main.py \
        --model_name LSTM_AE \
        --default_cfg ./configs/default_setting.yaml \
        --model_cfg   ./configs/model_setting.yaml \
        --opts \
            DATASET.dataname=$DS \
            TRAIN.epoch=10 \
            TRAIN.early_stopping_count=5 \
            DATASET.batch_size=128 \
            DATASET.stride_len=1 \
        2>&1 | tee -a $LOG
    echo ">>> LSTM_AE | $DS done | $(date)" | tee -a $LOG
done

# ── USAD ─────────────────────────────────────────────────────────────
for DS in "${DATASETS[@]}"; do
    echo "" | tee -a $LOG
    echo ">>> USAD | Dataset: $DS | $(date)" | tee -a $LOG
    echo "---" | tee -a $LOG
    python main.py \
        --model_name USAD \
        --default_cfg ./configs/default_setting.yaml \
        --model_cfg   ./configs/model_setting.yaml \
        --opts \
            DATASET.dataname=$DS \
            TRAIN.epoch=20 \
            TRAIN.lradj=none \
            TRAIN.early_stopping_count=9999 \
            OPTIMIZER.opt_name=adam \
            OPTIMIZER.lr=0.001 \
            OPTIMIZER.params.weight_decay=0.0 \
            DATASET.batch_size=512 \
        2>&1 | tee -a $LOG
    echo ">>> USAD | $DS done | $(date)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "========================================" | tee -a $LOG
echo "All evaluations done: $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG