from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn_data,
                  val_data,
                  tst_data,
                  scaler_type='standard',
                  ):
    
    # NaN 값을 전후값의 평균으로 보간 (데이터 정제)
    def interpolate_nan(data):
        """NaN 값을 전후값의 평균으로 보간"""
        data_copy = data.copy()
        for i in range(data_copy.shape[1]):  # 각 피처별로
            column = data_copy[:, i]
            nan_mask = np.isnan(column)
            
            if np.any(nan_mask):
                # 유효한 값들로 선형 보간
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) > 0:
                    # 양 끝의 NaN은 가장 가까운 유효값으로 대체
                    if len(valid_indices) == 1:
                        column[nan_mask] = column[valid_indices[0]]
                    else:
                        column = np.interp(np.arange(len(column)), valid_indices, column[valid_indices])
                else:
                    # 모든 값이 NaN이면 0으로 대체
                    column[nan_mask] = 0.0
            data_copy[:, i] = column
        return data_copy
    
    trn_data = interpolate_nan(trn_data)
    tst_data = interpolate_nan(tst_data)
    if val_data is not None:
        val_data = interpolate_nan(val_data)
    
    if scaler_type == 'standard' :
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    
    scaler.fit(trn_data)
    
    trn = scaler.transform(trn_data)
    tst = scaler.transform(tst_data)
    
    dev = None
    if val_data is not None:
        dev = scaler.transform(val_data)
    
    
    return trn, dev, tst