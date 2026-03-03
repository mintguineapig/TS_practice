for model_name in DLinear PatchTST iTransformer TimesNet Informer; do
    for dataname in ETTh1 ETTh2 ETTm1 ETTm2; do
        python main.py \
            --model_name $model_name \
            --default_cfg ./configs/default_setting.yaml \
            --model_cfg ./configs/model_setting.yaml \
            DATASET.dataname $dataname
    done
done
