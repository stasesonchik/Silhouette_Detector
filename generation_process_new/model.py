import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import ATT_CLASSES


# === Sinusoidal positional encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


# === DropPath / Stochastic Depth ===
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# === PreNorm Transformer Encoder Layer ===
class PreNormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path_rate=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        # self-attention
        y = self.norm1(x)
        y, _ = self.self_attn(y, y, y, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(self.dropout1(y))
        # feed-forward
        y = self.norm2(x)
        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        x = x + self.drop_path(self.dropout2(y))
        return x


# === Attention Pooling ===
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# === Main VisionAttrTransformer ===
class VisionAttrTransformer(nn.Module):
    def __init__(self,
                 hidden_dim=768,
                 num_heads=12,
                 num_layers=6,
                 attr_sizes=ATT_CLASSES,
                 drop_path_rate=0.1):
        super().__init__()
        self.input_proj = nn.Linear(3584, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dropout = nn.Dropout(0.2)

        # drop path rates per layer
        dpr = list(torch.linspace(0, drop_path_rate, num_layers).numpy())
        self.layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                drop_path_rate=dpr[i]
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = AttentionPooling(hidden_dim)

        # heads
        self.heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, size)
            ) for attr, size in attr_sizes.items()
        })

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # x: [B, N, P, D]
        B, N, P, D = x.size()
        x = x.view(B, N * P, D)
        x = self.input_proj(x) + self.pos_encoding(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            m = mask.unsqueeze(-1).expand(-1, -1, P).reshape(B, N * P)
            cls_m = torch.ones(B, 1, device=x.device, dtype=torch.bool)
            pad_mask = ~(torch.cat([cls_m, m], dim=1))
        else:
            pad_mask = None
            m = None

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        x = self.norm(x)
        cls_out = x[:, 0]
        seq_out = x[:, 1:]
        pooled = self.attention_pool(seq_out, m)
        final = cls_out + pooled

        return {attr: head(final) for attr, head in self.heads.items()}
