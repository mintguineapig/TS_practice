import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    paper link: https://arxiv.org/abs/2310.06625 (ICLR 2024)

    핵심 아이디어:
    - 기존 Transformer: 시간 축(time steps)을 token으로 처리
    - iTransformer: 변수 축(variates)을 token으로 처리 (Inverted)
    - 각 변수의 전체 시계열을 하나의 token으로 embedding
    - 변수 간 상관관계(inter-variate correlation)를 attention으로 학습
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Inverted Embedding: [B, T, N] → [B, N, d_model]
        # 각 변수(N)의 시계열(T)을 d_model 차원으로 embedding
        self.enc_embedding = DataEmbedding_inverted(
            c_in       = configs.seq_len,   # 시계열 길이를 입력 차원으로
            d_model    = configs.d_model,
            embed_type = configs.embed,
            freq       = configs.freq,
            dropout    = configs.dropout
        )

        # Encoder: 변수 간 attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag        = False,
                            factor           = configs.factor,
                            attention_dropout= configs.dropout,
                            output_attention = configs.output_attention
                        ),
                        d_model  = configs.d_model,
                        n_heads  = configs.n_heads
                    ),
                    d_model    = configs.d_model,
                    d_ff       = configs.d_ff,
                    dropout    = configs.dropout,
                    activation = configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer = nn.LayerNorm(configs.d_model)
        )

        # Projection: d_model → pred_len (각 변수별 미래 예측)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc:      [B, seq_len, N]   - 인코더 입력 데이터
        x_mark_enc: [B, seq_len, D]   - 인코더 타임스탬프
        x_dec:      [B, label_len+pred_len, N]  - 디코더 입력 (iTransformer는 사용 안 함)
        x_mark_dec: [B, label_len+pred_len, D]  - 디코더 타임스탬프 (사용 안 함)

        Returns:
            dec_out: [B, pred_len, N]
        """
        # Inverted Embedding: [B, seq_len, N] → [B, N, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark=None)

        # Encoder: 변수 간 attention
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection: [B, N, d_model] → [B, N, pred_len] → [B, pred_len, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        return dec_out
