"""Multivariate SEER head: cross-feature attention and one target output."""
from __future__ import annotations

import torch
import torch.nn as nn

from ts_benchmark.baselines.seer.layers.Embed import AugmentedEmbedding
from ts_benchmark.baselines.seer.layers.RevIN import RevIN
from ts_benchmark.baselines.seer.utils.masked_attention import FullAttention, AttentionLayer
from ..layers.Transformer_EncDec import Encoder, EncoderLayer


class MultivariateSEERModel(nn.Module):
    """SEER encoder with cross-variable attention and a single target head.

    The original adapter flattens batch-times-variables and processes every
    variable independently. This variant keeps the SEER patch pipeline, then
    applies attention across the eight variables at every patch position before
    selecting the NOX representation for a one-channel forecast.
    """

    def __init__(self, config):
        super().__init__()
        self.n_vars = config.enc_in
        self.target_index = config.target_index
        self.config = config
        self.revin = RevIN(self.n_vars)
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.patch_num = int((config.seq_len - self.patch_len) / self.stride + 2)

        self.augmented_embedding = AugmentedEmbedding(
            config,
            config.d_model,
            config.patch_len,
            config.stride,
            config.stride,
            config.dropout,
        )

        self.score_threshold = config.score_threshold
        if self.score_threshold != 0:
            self.casual_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                config.factor,
                                attention_dropout=config.dropout,
                                output_attention=config.output_attention,
                            ),
                            config.d_model,
                            config.n_heads,
                        ),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                ],
                norm_layer=nn.LayerNorm(config.d_model),
            )

        self.full_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )

        # Cross-variable attention is applied independently at every patch.
        self.cross_variable_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_variable_norm = nn.LayerNorm(config.d_model)

        self.head_nf = int(
            config.d_model
            * config.scaling
            * int((config.seq_len - config.patch_len) / config.stride + 2 + 1)
        )
        if config.scaling != 1:
            self.linear_projection = nn.Linear(
                config.d_model,
                int(config.d_model * config.scaling),
                bias=False,
            )

        self.target_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.head_dropout),
        )

        if self.score_threshold != 0:
            self.linear_filter = nn.Sequential(
                nn.Linear(config.d_model, 1),
                nn.Sigmoid(),
            )

    def replaced_embedding(self, enc_out, score):
        mask_zero = score != 0
        inv = torch.zeros_like(score)
        inv[mask_zero] = (1 / score[mask_zero]).detach()
        identity = score * inv
        raw_mask = identity * (score > self.score_threshold).float()
        raw_mask[:, 0] = 1
        series_token = enc_out[:, 0:1, :].expand(-1, self.patch_num + 1, -1)
        return enc_out * raw_mask.unsqueeze(-1) + series_token * (1 - raw_mask.unsqueeze(-1))

    def forward(self, inputs, input_mark):
        # RevIN normalizes each input variable using only the historical window.
        normalized = self.revin(inputs, "norm")
        x_enc = normalized.permute(0, 2, 1)
        enc_out, importance_loss = self.augmented_embedding(x_enc, input_mark)

        if self.score_threshold != 0:
            score = self.linear_filter(enc_out).squeeze(-1)
            enc_out = self.replaced_embedding(enc_out, score)
            enc_out, _ = self.casual_encoder(enc_out)
            enc_out, _ = self.full_encoder(enc_out)
        else:
            enc_out, _ = self.full_encoder(enc_out)

        batch_size, _, token_count, d_model = (
            inputs.shape[0],
            self.n_vars,
            enc_out.shape[-2],
            enc_out.shape[-1],
        )
        # Restore variables, attend across the 8 features at each token position.
        enc_out = enc_out.reshape(batch_size, self.n_vars, token_count, d_model)
        cross = enc_out.permute(0, 2, 1, 3).reshape(
            batch_size * token_count, self.n_vars, d_model
        )
        mixed, _ = self.cross_variable_attention(cross, cross, cross, need_weights=False)
        cross = self.cross_variable_norm(cross + mixed)
        enc_out = cross.reshape(
            batch_size, token_count, self.n_vars, d_model
        ).permute(0, 2, 1, 3)

        # Keep only the target variable representation and produce one output.
        target_features = enc_out[:, self.target_index]
        if self.config.scaling != 1:
            target_features = self.linear_projection(target_features)
        prediction = self.target_head(target_features).unsqueeze(-1)

        # Undo only the target channel's RevIN statistics.
        target_mean = self.revin.mean[:, :, self.target_index]
        target_stdev = self.revin.stdev[:, :, self.target_index]
        prediction = prediction * target_stdev.unsqueeze(-1) + target_mean.unsqueeze(-1)
        return prediction, importance_loss

