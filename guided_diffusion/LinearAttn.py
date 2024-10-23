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
    def __init__(self):
        super(FeedbackAttention, self).__init__()
        self.attn = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def build_attention_layers(self, input_channels):
        # Dynamically build attention layers based on the input channels
        return nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, current, feedback):
        input_channels = current.shape[1]
        # self.attn = self.build_attention_layers(input_channels)
        # Build attention layers only if they haven't been built or if input channels change
        if self.attn is None or self.attn[0].in_channels != input_channels * 2:
            self.attn = self.build_attention_layers(input_channels).to(self.device)
        combined = torch.cat([current, feedback], dim=1)
        attention_weights = self.attn(combined)
        return current * attention_weights + feedback * (1 - attention_weights)