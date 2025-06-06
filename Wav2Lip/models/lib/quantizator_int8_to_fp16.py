import torch
import torch.nn as nn
from torch.functional import F
from lib.config import Config


class QuantizedConv2d(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.bias = None
        self.scale = None
        self.device = Config.device

        self.quantize_weights(conv_layer.weight, conv_layer.bias)

    def quantize_weights(self, weights, bias):
        self.dtype = weights.dtype
        self.scale = torch.tensor(255.0, device=self.device, dtype=torch.float32)
        min_val = weights.min().item()
        max_val = weights.max().item()
        scale = (max_val - min_val) / 255.0
        q_weights = torch.round((weights - min_val) / scale).clamp(0, 255).to(torch.uint8).to(self.device).flatten()

        q_weights_1 = q_weights[0::2]
        q_weights_2 = q_weights[1::2]

        if bias is not None:
            self.bias = nn.Parameter(bias.to(self.device), requires_grad = False)
        else:
            self.bias = None
        self.register_buffer("packed_weight", (q_weights_1.to(torch.float16) + (q_weights_2.to(torch.float16) / self.scale)))
        self.register_buffer('scale_w', torch.tensor(scale, device=self.device, dtype=self.dtype))
        self.register_buffer("zero_point", torch.tensor(min_val, device=self.device, dtype=self.dtype))

    def unpack_weights(self):
        q1 = self.packed_weight.floor()
        q2 = ((self.packed_weight - q1) * self.scale).round().clamp(0, 255).to(torch.uint8)
        q1 = q1.to(torch.uint8)

        q_weights = torch.empty(q1.numel() * 2, dtype=torch.uint8, device=self.device)
        q_weights[0::2] = q1
        q_weights[1::2] = q2
        del q1, q2

        q_weights = q_weights.reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).to(torch.float16) * self.scale_w + self.zero_point
        return q_weights
    
    def forward(self, x):
        return F.conv2d(x, self.unpack_weights(), self.bias, self.stride, self.padding, self.dilation, self.groups)



class QuantizedConvTranspose2d(nn.Module):
    def __init__(self, conv_transpose_layer):
        super().__init__()
        self.in_channels = conv_transpose_layer.in_channels
        self.out_channels = conv_transpose_layer.out_channels
        self.kernel_size = conv_transpose_layer.kernel_size
        self.stride = conv_transpose_layer.stride
        self.padding = conv_transpose_layer.padding
        self.output_padding = conv_transpose_layer.output_padding
        self.dilation = conv_transpose_layer.dilation
        self.groups = conv_transpose_layer.groups
        self.bias = None
        self.scale = None
        self.device = Config.device

        self.quantize_weights(conv_transpose_layer.weight, conv_transpose_layer.bias)

    def quantize_weights(self, weights, bias):
        self.dtype = weights.dtype
        self.scale = torch.tensor(255.0, device=self.device, dtype=torch.float32)
        min_val = weights.min().item()
        max_val = weights.max().item()
        scale = (max_val - min_val) / 255.0
        q_weights = torch.round((weights - min_val) / scale).clamp(0, 255).to(torch.uint8).to(self.device).flatten()

        q_weights_1 = q_weights[0::2]
        q_weights_2 = q_weights[1::2]

        if bias is not None:
            self.bias = nn.Parameter(bias.to(self.device), requires_grad = False)
        else:
            self.bias = None
        self.register_buffer("packed_weight", (q_weights_1.to(torch.float16) + (q_weights_2.to(torch.float16) / self.scale)))
        self.register_buffer('scale_w', torch.tensor(scale, device=self.device, dtype=self.dtype))
        self.register_buffer("zero_point", torch.tensor(min_val, device=self.device, dtype=self.dtype))

    def unpack_weights(self):
        q1 = self.packed_weight.floor()
        q2 = ((self.packed_weight - q1) * self.scale).round().clamp(0, 255).to(torch.uint8)
        q1 = q1.to(torch.uint8)

        q_weights = torch.empty(q1.numel() * 2, dtype=torch.uint8, device=self.device)
        q_weights[0::2] = q1
        q_weights[1::2] = q2
        del q1, q2

        q_weights = q_weights.reshape(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]).to(torch.float16) * self.scale_w + self.zero_point
        return q_weights
    
    def forward(self, x):
        return F.conv_transpose2d(x, self.unpack_weights(), self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)