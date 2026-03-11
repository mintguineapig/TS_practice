import torch
import torch.nn as nn
import numpy as np


class USAD(nn.Module):
    """
    USAD: UnSupervised Anomaly Detection on Multivariate Time Series
    Paper: KDD 2020 (https://dl.acm.org/doi/10.1145/3394486.3403392)

    Architecture:
        - Shared Encoder  E  : W -> z
        - Decoder D1          : z -> W  (AE1 = E + D1)
        - Decoder D2          : z -> W  (AE2 = E + D2)

    Training (두 단계, 에폭 n / 전체 N):
        Phase 1 (AE1 최적화):
            L_AE1 = (1/n)||W - AE1(W)||² + (1 - 1/n)||W - AE2(AE1(W))||²
            → AE1은 재구성을 잘 하면서 동시에 AE2를 속이도록 학습

        Phase 2 (AE2 최적화):
            L_AE2 = (1/n)||W - AE2(W)||² - (1 - 1/n)||W - AE2(AE1(W))||²
            → AE2는 정상 데이터를 잘 재구성하면서 AE1의 출력을 구별하도록 학습

    Anomaly Score (추론):
        score = alpha * ||W - AE1(W)||² + beta * ||W - AE2(AE1(W))||²
        (기본값: alpha=1/n, beta=1-1/n where n=n_epochs)
    """

    def __init__(self, cfg):
        super(USAD, self).__init__()
        self.seq_len = cfg.seq_len          # 슬라이딩 윈도우 길이
        self.dim_in  = cfg.dim_in           # 입력 feature 수
        self.w_size  = cfg.seq_len * cfg.dim_in   # 입력 벡터 크기 (평탄화)
        self.z_size  = cfg.latent_size      # 잠재 공간 크기

        # ── Shared Encoder: W → z ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(self.w_size, self.w_size // 2),
            nn.ReLU(),
            nn.Linear(self.w_size // 2, self.z_size),
            nn.ReLU(),
        )

        # ── Decoder 1 (AE1 의 복원기): z → W ──────────────────────────────
        self.decoder1 = nn.Sequential(
            nn.Linear(self.z_size, self.w_size // 2),
            nn.ReLU(),
            nn.Linear(self.w_size // 2, self.w_size),
            nn.Sigmoid(),
        )

        # ── Decoder 2 (AE2 의 복원기): z → W ──────────────────────────────
        self.decoder2 = nn.Sequential(
            nn.Linear(self.z_size, self.w_size // 2),
            nn.ReLU(),
            nn.Linear(self.w_size // 2, self.w_size),
            nn.Sigmoid(),
        )

    # ── 내부 편의 메서드 ────────────────────────────────────────────────────
    def _encode(self, w):
        return self.encoder(w)

    def _decode1(self, z):
        return self.decoder1(z)

    def _decode2(self, z):
        return self.decoder2(z)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, input, input_timestamp, target, criterion,
                cal_score=False, epoch=1, n_epochs=1, phase=None):
        """
        Parameters
        ----------
        input          : (B, S, F) 입력 시계열 윈도우
        input_timestamp: 사용하지 않음 (인터페이스 호환)
        target         : (B, S, F) 재구성 대상 (= input)
        criterion      : 사용하지 않음 (손실은 내부 정의)
        cal_score      : True 이면 이상 점수도 반환
        epoch          : 현재 에폭 (1-indexed)
        n_epochs       : 전체 에폭 수
        phase          : 1 → AE1 손실만, 2 → AE2 손실만, None → cal_score 모드

        Returns
        -------
        phase=1  : (decoded, loss_ae1)
        phase=2  : (decoded, loss_ae2)
        cal_score: (decoded, reconstruction_loss, score)
        """
        B, S, F = input.size()

        # [B, S, F] → [B, S*F]  (입력을 평탄화)
        w = input.reshape(B, -1)

        # ── 공유 인코더 통과 ───────────────────────────────────────────────
        z = self._encode(w)

        # ── AE1(W) = D1(E(W)) ────────────────────────────────────────────
        w_hat1 = self._decode1(z)

        # ── 에폭 가중치 n ─────────────────────────────────────────────────
        n = epoch  # 1 ≤ n ≤ n_epochs

        # ── Phase 1: AE1 손실 ─────────────────────────────────────────────
        if phase == 1:
            # AE2(AE1(W)) = D2(E(D1(E(W))))  ← AE1 속임 항 계산
            w_hat12 = self._decode2(self._encode(w_hat1))

            loss_ae1 = (
                (1 / n)       * torch.mean((w - w_hat1)  ** 2) +
                (1 - 1 / n)   * torch.mean((w - w_hat12) ** 2)
            )
            decoded = w_hat1.reshape(B, S, F)
            return decoded, loss_ae1

        # ── Phase 2: AE2 손실 ─────────────────────────────────────────────
        if phase == 2:
            # AE2(W)      = D2(E(W))
            w_hat2  = self._decode2(z)
            # AE2(AE1(W)) = D2(E(AE1(W)))  ← AE2 판별 항 계산
            w_hat12 = self._decode2(self._encode(w_hat1.detach()))  # AE1 그래프 분리

            loss_ae2 = (
                (1 / n)       * torch.mean((w - w_hat2)  ** 2) -
                (1 - 1 / n)   * torch.mean((w - w_hat12) ** 2)
            )
            decoded = w_hat2.reshape(B, S, F)
            return decoded, loss_ae2

        # ── 추론 / 이상 점수 계산 ─────────────────────────────────────────
        # cal_score=True 시 혹은 phase가 지정되지 않은 경우
        w_hat2  = self._decode2(z)
        w_hat12 = self._decode2(self._encode(w_hat1))

        recon_loss = torch.mean((w - w_hat1) ** 2)

        if cal_score:
            score = self.cal_anomaly_score(w, w_hat1, w_hat12, n_epochs)
            decoded = w_hat1.reshape(B, S, F)
            return decoded, recon_loss, score

        # 단순 평가 모드 (val)
        decoded = w_hat1.reshape(B, S, F)
        return decoded, recon_loss

    # ── 이상 점수 ────────────────────────────────────────────────────────────
    def cal_anomaly_score(self, w, w_hat1, w_hat12, n_epochs):
        """
        score = alpha * ||W - AE1(W)||² + beta * ||W - AE2(AE1(W))||²

        alpha = 1/n_epochs,  beta = 1 - 1/n_epochs
        (에폭이 클수록 AE2 판별 항의 비중이 커짐)
        """
        alpha = 1 / n_epochs
        beta  = 1 - 1 / n_epochs

        score1  = torch.mean((w - w_hat1)  ** 2, dim=1)  # [B]
        score12 = torch.mean((w - w_hat12) ** 2, dim=1)  # [B]

        score = alpha * score1 + beta * score12  # [B]
        return score.detach().cpu().numpy().reshape(-1, 1)
