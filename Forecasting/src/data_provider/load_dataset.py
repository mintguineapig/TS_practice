from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from utils.timefeatures import time_features_from_date

def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    time_embedding: list = [True, 'h'],
    del_feature: list = None
):
    df = pd.read_csv(os.path.join(datadir, dataname + '.csv'))
    n = len(df)
    idx_trn, idx_val = int(n * split_rate[0]), int(n * (split_rate[0] + split_rate[1]))

    # 데이터셋 준비    
    ts = np.array(time_features_from_date(df['date'], int(time_embedding[0]), time_embedding[1])).astype(np.float32)
    data = df.drop(columns=['date'] + (del_feature or [])).values.astype(np.float32)
    
    # 분할
    trn, val, tst = data[:idx_trn], data[idx_trn : idx_val], data[idx_val: ] 
    trn_ts, val_ts, tst_ts = ts[:idx_trn], ts[idx_trn : idx_val], ts[idx_val: ] 
    var = data.shape[1]
     
    return trn, trn_ts, val, val_ts, tst, tst_ts, var