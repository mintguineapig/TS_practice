#!/bin/bash
# =============================================================
#  LSTM_AE  w/ RevIN  vs  w/o RevIN  비교 평가 스크립트
#  실행: bash run_revin_eval.sh
# =============================================================

export CUDA_VISIBLE_DEVICES=0
LOG_DIR="./eval_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " [1/2] LSTM_AE  WITHOUT RevIN"
echo "============================================================"
python main.py \
    --model_name LSTM_AE \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg   ./configs/model_setting_no_revin.yaml \
    --opts DEFAULT.exp_name=LSTM_AE_no_revin \
2>&1 | tee "$LOG_DIR/lstm_ae_no_revin.log"

echo ""
echo "============================================================"
echo " [2/2] LSTM_AE  WITH RevIN"
echo "============================================================"
python main.py \
    --model_name LSTM_AE \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg   ./configs/model_setting.yaml \
    --opts DEFAULT.exp_name=LSTM_AE_revin \
2>&1 | tee "$LOG_DIR/lstm_ae_revin.log"

echo ""
echo "============================================================"
echo " 📊  결과 비교"
echo "============================================================"
python compare_revin.py \
    --log_no_revin "$LOG_DIR/lstm_ae_no_revin.log" \
    --log_revin    "$LOG_DIR/lstm_ae_revin.log"
