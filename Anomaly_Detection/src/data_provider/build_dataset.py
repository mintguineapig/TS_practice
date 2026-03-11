import torch
from torch.utils.data import Dataset
import numpy as np

class BuildDataset(Dataset):
    """
    시계열 데이터를 sliding window 방식으로 처리하는 Dataset 클래스
    """
    
    def __init__(self, data, timestamps=None, labels=None, seq_len=100, stride_len=1):
        """
        Args:
            data: 시계열 데이터 (numpy array) [time_steps, features]
            timestamps: 타임스탬프 (선택사항)
            labels: 레이블 데이터 (테스트용, 선택사항)
            seq_len: 시퀀스 길이
            stride_len: 슬라이딩 윈도우 스트라이드
        """
        self.data = data
        self.timestamps = timestamps
        self.labels = labels
        self.seq_len = seq_len
        self.stride_len = stride_len
        
        # 시퀀스 인덱스 계산
        self.indices = self._calculate_indices()
    
    def _calculate_indices(self):
        """슬라이딩 윈도우 인덱스 계산"""
        max_start = len(self.data) - self.seq_len + 1
        indices = list(range(0, max_start, self.stride_len))
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: [seq_len, features] 시계열 시퀀스
            label: 레이블 (있는 경우) - 이상탐지에서는 해당 시퀀스의 마지막 포인트 레이블
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        sequence = torch.FloatTensor(self.data[start_idx:end_idx])
        
        if self.labels is not None:
            # 이상탐지에서는 시퀀스의 마지막 시점 레이블 사용
            label = torch.LongTensor([self.labels[end_idx - 1]])
            return sequence, label
        else:
            return sequence