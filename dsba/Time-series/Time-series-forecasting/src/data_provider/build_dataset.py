from torch.utils.data import Dataset
import numpy as np
import torch

class BuildDataset(Dataset):
    def __init__(self, data, data_ts, seq_len, label_len, pred_len):
        self.data = np.array(data, dtype = np.float32)
        self.data_ts = np.array(data_ts, dtype = np.float32)
        self.seq_len = seq_len   # 모델이 보는 과거 길이
        self.label_len = label_len
        self.pred_len = pred_len
        self.valid_window = len(data) - seq_len - pred_len + 1
        return None
    
    # 시작 idx, idx 번째 슬라이딩 윈도우 데이터를 잘라서 반환
    def __getitem__(self, idx): 
        input_start = idx  
        input_end = idx + self.seq_len
        target_start = input_end - self.label_len
        target_end = input_end + self.pred_len
        
        item = {'input_data' : torch.tensor(self.data[input_start : input_end]),
                'target_data' : torch.tensor(self.data[target_start : target_end]),
                'input_ts' : torch.tensor(self.data_ts[input_start : input_end]),
                'target_ts' : torch.tensor(self.data_ts[target_start : target_end]) }
        return item   
    
    def __len__(self):        
        return self.valid_window
