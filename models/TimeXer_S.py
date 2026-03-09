import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B_original = cross.shape[0]
        D = cross.shape[-1]

        # Self-Attention
        identity = x
        x = identity + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # Cross-Attention on Global Token
        x_glb_query = x[:, -1, :].unsqueeze(1)
        x_glb_query_reshaped = torch.reshape(x_glb_query, (B_original, -1, D))

        x_glb_attn_output_reshaped = self.dropout(self.cross_attention(
            x_glb_query_reshaped, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
        )[0])

        x_glb_attn_output = torch.reshape(x_glb_attn_output_reshaped, (x_glb_query.shape[0], 1, D))
        updated_x_glb = self.norm2(x_glb_query + x_glb_attn_output)

        # FFN Block
        patches_only = x[:, :-1, :]
        ffn_input = torch.cat([patches_only, updated_x_glb], dim=1)

        ffn_output = self.dropout(self.activation(self.conv1(ffn_input.transpose(-1, 1))))
        ffn_output = self.dropout(self.conv2(ffn_output).transpose(-1, 1))

        return self.norm3(ffn_input + ffn_output)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)

        self.n_vars_en_embed = 1 if self.features in ['S', 'MS'] else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars_en_embed, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding_hist = DataEmbedding_inverted(self.seq_len, configs.d_model, configs.embed, configs.freq,
                                                        configs.dropout)
        self.ex_embedding_aug = DataEmbedding_inverted(self.seq_len + self.pred_len, configs.d_model, configs.embed,
                                                       configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_future_exog, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        target_series = x_enc[:, :, -1].unsqueeze(-1)
        en_embed, n_vars = self.en_embedding(target_series.permute(0, 2, 1))

        x_exog_hist = x_enc[:, :, :-1]
        N_exog_hist = x_exog_hist.shape[-1]
        use_future_exog = (x_future_exog is not None and x_future_exog.shape[-1] == N_exog_hist and N_exog_hist > 0)

        if use_future_exog:
            x_exog_combined = torch.cat([x_exog_hist, x_future_exog], dim=1)
            x_mark_combined = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
            cross_embed = self.ex_embedding_aug(x_exog_combined, x_mark_combined)
        else:
            cross_embed = self.ex_embedding_hist(x_exog_hist, x_mark_enc)

        enc_out = self.encoder(en_embed, cross_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            target_stdev = stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            target_means = means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * target_stdev + target_means

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding_hist(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_future_exog=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                if x_mark_dec is None:
                    F_time = x_mark_enc.shape[-1] if x_mark_enc is not None else 0
                    x_mark_dec = torch.zeros(x_enc.shape[0], self.pred_len, F_time, device=x_enc.device)
                dec_out = self.forecast(x_enc, x_mark_enc, x_future_exog, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None