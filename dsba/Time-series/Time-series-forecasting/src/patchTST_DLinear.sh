#!/bin/bash
# ================================================================
# patchTST_DLinear.sh
#
# Runs PatchTST and DLinear on ETTh1/ETTh2/ETTm1/ETTm2 using
# dataset-specific hyperparameters from the official papers.
#
# PatchTST (ICLR 2023): different model size per dataset group
#   ETTh1/ETTh2  →  d_model=16,  n_heads=4,  d_ff=128, dropout=0.3
#   ETTm1/ETTm2  →  d_model=128, n_heads=16, d_ff=256, dropout=0.2
#
# LR scheduler per paper:
#   ETTm1/ETTm2 → lradj=TST (OneCycleLR, pct_start=0.4, patience=20)
#   ETTh1/ETTh2 → lradj=type3 (paper default), patience=5
#
# DLinear (AAAI 2023): same hyperparameters across all datasets
#
# Prediction horizons: T = 96, 192, 336, 720  (all four from the paper)
# ================================================================

DEFAULT_CFG=./configs/default_setting.yaml
MODEL_CFG=./configs/patchTST_DLinear.yaml

echo "========================================================"
echo " PatchTST + DLinear  (paper hyperparameters)"
echo " Config  : ${MODEL_CFG}"
echo " pred_len: 96 192 336 720"
echo " Started : $(date)"
echo "========================================================"

# ----------------------------------------------------------------
# PatchTST on ETTh1/ETTh2
#   lradj=type3 (constant lr for <10 epochs, then 0.1x — paper default)
#   patience=5
# ----------------------------------------------------------------
for pred_len in 96 192 336 720; do
    for dataname in ETTh1 ETTh2; do
        model_key="PatchTST_${dataname}"
        echo ""
        echo ">>> [PatchTST] dataset=${dataname}  pred_len=${pred_len}  lradj=type3"
        python main.py \
            --model_name  "${model_key}" \
            --default_cfg "${DEFAULT_CFG}" \
            --model_cfg   "${MODEL_CFG}" \
            DATASET.dataname "${dataname}" \
            DATASET.pred_len "${pred_len}" \
            TRAIN.lradj type3 \
            TRAIN.early_stopping_count 5
    done
done

# ----------------------------------------------------------------
# PatchTST on ETTm1/ETTm2
#   lradj=TST (OneCycleLR, pct_start=0.4 — paper setting)
#   patience=20
# ----------------------------------------------------------------
for pred_len in 96 192 336 720; do
    for dataname in ETTm1 ETTm2; do
        model_key="PatchTST_${dataname}"
        echo ""
        echo ">>> [PatchTST] dataset=${dataname}  pred_len=${pred_len}  lradj=TST"
        python main.py \
            --model_name  "${model_key}" \
            --default_cfg "${DEFAULT_CFG}" \
            --model_cfg   "${MODEL_CFG}" \
            DATASET.dataname "${dataname}" \
            DATASET.pred_len "${pred_len}" \
            TRAIN.lradj TST \
            TRAIN.pct_start 0.4 \
            TRAIN.early_stopping_count 20
    done
done

# ----------------------------------------------------------------
# DLinear
# Same hyperparameters for all four ETT datasets.
# ----------------------------------------------------------------
for pred_len in 96 192 336 720; do
    for dataname in ETTh1 ETTh2 ETTm1 ETTm2; do
        echo ""
        echo ">>> [DLinear] dataset=${dataname}  pred_len=${pred_len}"
        python main.py \
            --model_name  DLinear \
            --default_cfg "${DEFAULT_CFG}" \
            --model_cfg   "${MODEL_CFG}" \
            DATASET.dataname "${dataname}" \
            DATASET.pred_len "${pred_len}"
    done
done

echo ""
echo "========================================================"
echo " All experiments finished: $(date)"
echo "========================================================"
