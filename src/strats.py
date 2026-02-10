# Alexander Ye 

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x): # x is (batch_size, max_len)
        x = torch.unsqueeze(x, -1) # (batch_size, max_len, 1)
        x = x @ self.W1 + self.b1[None, None, :]
        x = self.activation(x)
        x = x @ self.W2
        return x

# Example
# args = Namespace(
#     num_layers=6,
#     hid_dim=512,
#     num_heads=8,
#     dropout=0.1,
#     attention_dropout=0.1,
#     V=?!
# )

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.num_layers
        self.d = args.hid_dim
        self.dff = self.d * 2
        self.attention_dropout = args.attention_dropout
        self.dropout = args.dropout
        self.h = args.num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h

        self.layers = nn.ModuleList([
            TransformerBlock(self.d, self.h, self.dff, self.dropout, self.attention_dropout)
            for _ in range(self.N)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

        

class TransformerBlock(nn.Module):
    def __init__(self,  d, num_heads, dff, dropout, attention_dropout):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.dff = dff
        self.head_size = d // num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        # attention
        self.query = nn.Linear(self.d, self.d, bias=False)
        self.key = nn.Linear(self.d, self.d, bias=False)
        self.value = nn.Linear(self.d, self.d, bias=False)
        self.projection = nn.Linear(d, d, bias=False)

        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        self.W1 = nn.Linear(self.d, self.dff, bias=True)
        self.W2 = nn.Linear(self.dff, self.d, bias=True)

    def forward(self, x, mask):
        B, T, C = x.size() # bsz, max_len, d

        # single layer handles all heads
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, h, T, dk)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, h, T, dk)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, h, T, dk)

        A = q @ k.transpose(-2, -1)  # (B, h, T, T)
        
        # Apply mask
        mask_2d = mask[:, :, None] * mask[:, None, :]  # (B, T, T)
        mask_2d = (1 - mask_2d)[:, None, :, :] * torch.finfo(x.dtype).min  # (B, 1, T, T)
        A = A + mask_2d

        if self.training:
            dropout_mask = (torch.rand_like(A) < self.attention_dropout).float() * torch.finfo(x.dtype).min
            A = A + dropout_mask
        
        A = torch.softmax(A, dim=-1)  # (B, h, T, T)
        
        # Apply attention
        out = A @ v  # (B, h, T, dk)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous()  # (B, T, h, dk)
        out = out.view(B, T, self.d)  # (B, T, d)
        
        # Output projection
        out = self.projection(out)
        out = F.dropout(out, self.dropout, self.training)

        x = self.norm1(out + x)

        ffn_out = self.W1(x)
        ffn_out = F.gelu(ffn_out)
        ffn_out = self.W2(ffn_out)
        ffn_out = F.dropout(ffn_out, self.dropout, self.training)
        
        x = self.norm2(ffn_out + x) 
        
        return x
        
class STraTS(nn.Module):
    def __init__(self, args):
        super().__init__(args)
        self.time_embd = CVE(args)
        self.value_embd = CVE(args)
        self.var_embd = nn.Embedding(args.V, args.hid_dim)
        self.transformer = Transformer(args)
        # self.fusion_attn =
        self.dropout = args.dropout
        self.V = args.V

    def forward(self, values, times, vars, obs_mask, demo):
        bsz, max_obs = values.size()

        demo_embd = self.demo_emb(demo)

        time_embd = self.time_embd(times)
        value_embd = self.value_embd(values)
        vari_embd = self.var_embd(vars)
        triplet_embd = time_embd+value_embd+vari_embd
        triplet_embd = F.dropout(triplet_embd, self.dropout, self.training)
        contextual_emb = self.transformer(triplet_embd, obs_mask) 

        # fusion self attention?

        ts_embd = (triplet_embd*attention_weights).sum(dim=1)
        ts_demo_embd = torch.cat((ts_embd, demo_embd), dim=-1)

        logits = self.binary_head(self.forecast_head(ts_demo_embd))[:,0]
        return F.sigmoid(logits)