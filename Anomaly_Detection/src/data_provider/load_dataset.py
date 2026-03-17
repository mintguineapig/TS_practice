import os
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(datadir: str,
                 dataname: str,
                 time_embedding: Optional[list] = None,
                 del_feature: Optional[list] = None
                 ):

    data_path = os.path.join(datadir, dataname)
    trn_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    tst_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    tst_label_df = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

    # 타임스탬프
    trn_ts = trn_df.iloc[:, 0].values
    tst_ts = tst_df.iloc[:, 0].values

    # sensor data
    trn_data = trn_df.iloc[:, 1:].values.astype(np.float32)
    tst_data = tst_df.iloc[:, 1:].values.astype(np.float32)

    # Label of tst data
    label = tst_label_df.iloc[:, 1].values.astype(np.float32)

    # 피처 수 계산 (del_feature 적용 전)
    var = trn_data.shape[1]

    # 특정 feature 삭제
    if del_feature:
        keep_indices = [i for i in range(var) if i not in del_feature]
        trn_data = trn_data[:, keep_indices]
        tst_data = tst_data[:, keep_indices]
        var = len(keep_indices)

    # train/validation 분할 (shuffle=False: 시계열 순서 유지)
    trn, val, trn_ts_split, val_ts_split = train_test_split(
        trn_data, trn_ts, test_size=0.2, random_state=42, shuffle=False
    )

    # 시간 특성 추출 (time_embedding이 지정된 경우)
    if time_embedding and len(time_embedding) > 0:
        try:
            trn_datetime = pd.to_datetime(trn_ts_split, unit='s')
            val_datetime = pd.to_datetime(val_ts_split, unit='s')
            tst_datetime = pd.to_datetime(tst_ts, unit='s')
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            try:
                trn_datetime = pd.to_datetime(trn_ts_split, unit='ms')
                val_datetime = pd.to_datetime(val_ts_split, unit='ms')
                tst_datetime = pd.to_datetime(tst_ts, unit='ms')
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                try:
                    base_date = pd.Timestamp('2000-01-01')
                    trn_datetime = base_date + pd.to_timedelta(trn_ts_split, unit='min')
                    val_datetime = base_date + pd.to_timedelta(val_ts_split, unit='min')
                    tst_datetime = base_date + pd.to_timedelta(tst_ts, unit='min')
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    # 모든 변환 실패 시 더미 시간 특성 생성
                    trn_time_features = np.zeros((len(trn), 1), dtype=np.float32)
                    val_time_features = np.zeros((len(val), 1), dtype=np.float32)
                    tst_time_features = np.zeros((len(tst_data), 1), dtype=np.float32)
                    trn = np.concatenate([trn, trn_time_features], axis=1)
                    val = np.concatenate([val, val_time_features], axis=1)
                    tst_data = np.concatenate([tst_data, tst_time_features], axis=1)
                    var = trn.shape[1]
                    return trn, trn_ts_split, val, val_ts_split, tst_data, tst_ts, var, label

        trn_time_features = time_features_from_date(trn_datetime, timeenc=0, freq='H')
        val_time_features = time_features_from_date(val_datetime, timeenc=0, freq='H')
        tst_time_features = time_features_from_date(tst_datetime, timeenc=0, freq='H')

        if hasattr(trn_time_features, 'values'):
            trn_time_features = trn_time_features.values.astype(np.float32)
            val_time_features = val_time_features.values.astype(np.float32)
            tst_time_features = tst_time_features.values.astype(np.float32)

        max_features = trn_time_features.shape[1]
        valid_indices = [i for i in time_embedding if isinstance(i, int) and 0 <= i < max_features]

        if valid_indices:
            trn_time_features = trn_time_features[:, valid_indices]
            val_time_features = val_time_features[:, valid_indices]
            tst_time_features = tst_time_features[:, valid_indices]
        else:
            trn_time_features = trn_time_features[:, :1]
            val_time_features = val_time_features[:, :1]
            tst_time_features = tst_time_features[:, :1]

        trn = np.concatenate([trn, trn_time_features], axis=1)
        val = np.concatenate([val, val_time_features], axis=1)
        tst_data = np.concatenate([tst_data, tst_time_features], axis=1)
        var = trn.shape[1]

    return trn, trn_ts_split, val, val_ts_split, tst_data, tst_ts, var, label

