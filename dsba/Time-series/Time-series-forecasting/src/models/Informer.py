import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
from layers.Transformer_EncDec import (
    Encoder, EncoderLayer, ConvLayer,
    Decoder, DecoderLayer
)


class Informer(nn.Module):
    """
    Informer: Efficient Transformer for Long Sequence Time-Series Forecasting
    paper link: https://arxiv.org/abs/2012.07436 (AAAI 2021 Best Paper)

    핵심 아이디어:
    - ProbSparse Self-Attention: O(L log L) 복잡도로 full attention 근사
    - Encoder의 self-distilling: ConvLayer로 시퀀스 길이를 절반씩 줄여 메모리 절약
    - Generative Decoder: 미래 시퀀스를 한 번에 생성 (auto-regressive 없음)
    """

    def __init__(self, configs):
        super(Informer, self).__init__()
        self.seq_len   = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len  = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # Encoder: ProbSparse Attention + self-distilling (ConvLayer)
        # ConvLayer 개수는 e_layers - 1 개 (마지막 레이어에는 적용 안 함)
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            conv_layers=[
                ConvLayer(configs.d_model)
                for _ in range(configs.e_layers - 1)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Decoder: Masked ProbSparse (self) + Full Attention (cross)
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        ProbAttention(
                            mask_flag=True,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    cross_attention=AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Args:
            x_enc      : [B, seq_len, enc_in]          encoder input
            x_mark_enc : [B, seq_len, time_features]   encoder time stamp
            x_dec      : [B, label_len+pred_len, dec_in] decoder input (hint + zeros)
            x_mark_dec : [B, label_len+pred_len, time_features]

        Returns:
            dec_out    : [B, pred_len, c_out]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
