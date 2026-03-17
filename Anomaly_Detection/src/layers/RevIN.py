import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        """
        Reversible Instance Normalization (Kim et al., ICLR 2022)

        Args:
            num_features  : 피처(채널) 수
            eps           : 수치 안정성을 위한 소수값
            affine        : True이면 학습 가능한 γ, β 파라미터 사용
            subtract_last : True이면 평균 대신 마지막 시점 값으로 정규화
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x    : [B, T, F] 형태의 입력 텐서
            mode : 'norm' (정규화) 또는 'denorm' (역정규화)
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"mode='{mode}' 은 지원하지 않습니다. 'norm' 또는 'denorm' 을 사용하세요.")
        return x

    def _init_params(self):
        """학습 가능한 affine 파라미터 γ, β 초기화 (shape: [F])"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))   # γ
        self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))  # β

    def _get_statistics(self, x: torch.Tensor):
        """인스턴스(샘플)별 평균·표준편차 계산 및 저장"""
        # x: [B, T, F] → 시간 축(dim=1) 기준으로 통계량 계산
        dim2reduce = tuple(range(1, x.ndim - 1))  # (1,) for 3-D input

        if self.subtract_last:
            # 마지막 시점 값을 기준점으로 사용
            self.mean = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """정규화: (x - mean) / stdev, 이후 affine 변환"""
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """역정규화: affine 역변환 후 원래 스케일로 복원"""
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        x = x * self.stdev + self.mean
        return x
