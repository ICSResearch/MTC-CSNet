import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class PostNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.norm(self.net(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=16 ** 2, num_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProduct()
        self.linear_out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, x):
        # linear projection
        key = self._reshape_to_heads(self.linear_k(x))
        value = self._reshape_to_heads(self.linear_v(x))
        query = self._reshape_to_heads(self.linear_q(x))

        scale = key.size(-1) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale)

        context = self._reshape_from_heads(context)

        output = self.linear_out(context)
        output = self.dropout(output)
        return output, attention


class EncodingLayer(nn.Module):
    def __init__(self, dim=16 ** 2, heads=8, dropout=0.):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.ffn = FeedForward(dim=dim, hidden_dim=int(dim * 0.5), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        context, attn = self.attn(x, x, x)[0], self.attn(x, x, x)[1]
        x = self.norm1(context + x)
        ffn = self.ffn(x)
        x = self.norm2(ffn + x)
        return x, attn


class IterLayers(nn.Module):
    def __init__(self, dim=16 ** 2, depth=12, heads=8, dropout=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(EncodingLayer(dim=dim, heads=heads, dropout=dropout))

    def forward(self, x):
        size = int(self.dim ** 0.5)
        x = Rearrange('b l h w -> b l (h w)')(x)
        x = x * math.sqrt(self.dim)
        attns = []
        for _ in range(self.depth):
            x, attn = self.layers[_](x)
            attns.append(attn)
        x = Rearrange('b l (h w)-> b l h w', h=size, w=size)(x)
        return x, attns
