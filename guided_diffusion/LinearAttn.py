import math

import torch

import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(channels // num_heads)

        # Linear projections for query, key, and value
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)

        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # Split into query, key, and value

        # Reshape for multi-head attention
        q, k, v = map(lambda t: t.reshape(b, self.num_heads, -1, h * w), qkv)

        # Linear attention: approximate softmax(QK^T) with kernel functions
        k = k.softmax(dim=-1)
        q = q.softmax(dim=-2)

        # Compute the attention map (approximated as linear dot products)
        context = torch.einsum('bhcn,bhct->bhnt', k, v)
        out = torch.einsum('bhct,bhnt->bhcn', q, context)

        # Reshape back to the original size
        out = out.reshape(b, c, h, w)
        return self.proj_out(out)

class FeedbackAttention(nn.Module):
    def __init__(self, input_channels):
        super(FeedbackAttention, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.attn = self.build_attention_layers(input_channels).to(self.device)

    def build_attention_layers(self, input_channels):
        # Dynamically build attention layers based on the input channels
        return nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, current, feedback):
        combined = torch.cat([current, feedback], dim=1)
        attention_weights = self.attn(combined)
        return current * attention_weights + feedback * (1 - attention_weights)


import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionModule(nn.Module):
    def __init__(self, channels):
        super(GatedFusionModule, self).__init__()
        # Linear layers to learn the gating mechanism
        self.gate_common = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()  # Output gate for common features
        )
        self.gate_distinct = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()  # Output gate for distinct features
        )
        # Combine fused output with another conv layer
        self.fuse_conv = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)

    def forward(self, common_features, distinct_features):
        # Compute gating weights
        gate_common_weight = self.gate_common(common_features)
        gate_distinct_weight = self.gate_distinct(distinct_features)

        # Weighted combination of features
        fused_common = common_features * gate_common_weight
        fused_distinct = distinct_features * gate_distinct_weight

        # Concatenate the fused features
        fused_features = torch.cat([fused_common, fused_distinct], dim=1)

        # Apply a final convolution to get the fused result
        output = self.fuse_conv(fused_features)

        return output
