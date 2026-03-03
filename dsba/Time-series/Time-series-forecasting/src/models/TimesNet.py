import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    FFT를 이용해 시계열에서 상위 k개의 주요 주기를 찾는 함수
    x: [B, T, C]
    returns: period list, frequency amplitudes
    """
    # FFT: 시간 축(dim=1)에 대해 주파수 성분 추출
    xf = torch.fft.rfft(x, dim=1)

    # 주파수 진폭의 평균 (배치, 채널 평균)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # DC 성분(주파수 0) 제거

    # 상위 k개 주파수 인덱스 추출
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 주파수 → 주기 변환: period = T / frequency
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesNet의 핵심 블록
    1D 시계열 → 2D (period × frequency) 변환 후 2D Conv 적용
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k  # 상위 k개 주기

        # 2D Conv: Inception 구조로 다양한 커널 크기 활용
        self.conv = nn.Sequential(
            Inception_Block_V1(
                in_channels  = configs.d_model,
                out_channels = configs.d_ff,
                num_kernels  = configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                in_channels  = configs.d_ff,
                out_channels = configs.d_model,
                num_kernels  = configs.num_kernels
            )
        )

    def forward(self, x):
        """
        x: [B, T, d_model]  (T = seq_len + pred_len)
        """
        B, T, N = x.size()

        # 1. FFT로 상위 k개 주기 탐지
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 2. 1D → 2D 변환: 시계열을 (period × ceil(T/period)) 행렬로 reshape
            # 길이가 딱 나눠지지 않으면 패딩
            if (T) % period != 0:
                length = (T // period + 1) * period
                padding = torch.zeros(B, length - T, N).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x

            # [B, length, N] → [B, N, ceil(T/period), period] (2D 형태)
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()
            # out: [B, N(=d_model), freq, period]

            # 3. 2D Conv 적용
            out = self.conv(out)
            # out: [B, d_model, freq, period]

            # 4. 2D → 1D 변환
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # 원래 길이로 자르기
            res.append(out[:, :T, :])

        # 5. 주기별 가중 합산 (Adaptive Aggregation)
        res = torch.stack(res, dim=-1)  # [B, T, N, k]

        # 주기 가중치 softmax 후 weighted sum
        period_weight = F.softmax(period_weight, dim=1)  # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).expand_as(res)
        res = torch.sum(res * period_weight, -1)  # [B, T, N]

        # Residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
    paper link: https://arxiv.org/abs/2210.02186 (ICLR 2023)

    핵심 아이디어:
    - 1D 시계열의 복잡한 시간적 변화를 2D 공간으로 변환
    - FFT로 주요 주기(period)를 찾아 (period × frequency) 2D 행렬로 reshape
    - 2D Conv(Inception)으로 intra-period(주기 내) + inter-period(주기 간) 패턴 동시 학습
    """

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # TimesBlock을 e_layers개 쌓음
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])

        # Embedding: [B, T, enc_in] → [B, T, d_model]
        self.enc_embedding = DataEmbedding(
            c_in       = configs.enc_in,
            d_model    = configs.d_model,
            embed_type = configs.embed,
            freq       = configs.freq,
            dropout    = configs.dropout
        )

        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Projection: d_model → c_out (예측값 출력)
        self.predict_linear = nn.Linear(configs.seq_len, configs.pred_len + configs.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc:      [B, seq_len, enc_in]
        x_mark_enc: [B, seq_len, D]
        x_dec:      [B, label_len+pred_len, enc_in]  (TimesNet은 사용 안 함)
        x_mark_dec: [B, label_len+pred_len, D]       (사용 안 함)

        Returns:
            dec_out: [B, pred_len, c_out]
        """
        # Normalization (RevIN 대신 간단한 mean normalization)
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding: [B, seq_len, enc_in] → [B, seq_len, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # seq_len → pred_len+seq_len으로 선형 확장 (예측 구간 포함)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # enc_out: [B, seq_len+pred_len, d_model]

        # TimesBlock 반복 적용
        for layer in self.model:
            enc_out = self.layer_norm(layer(enc_out))

        # Projection: [B, seq_len+pred_len, d_model] → [B, seq_len+pred_len, c_out]
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)

        # 마지막 pred_len만 반환
        return dec_out[:, -self.pred_len:, :]
