import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from lib.config import Config

class QuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, pca_variance=Config.pca_accuracy):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.pca_variance = pca_variance
        self.device = conv_layer.weight.device
        self.source_elements = conv_layer.weight.numel()

        # Распакуем веса в матрицу: (out_channels, in_channels * kH * kW)
        w = conv_layer.weight.detach().cpu()
        weight_matrix = w.view(w.size(0), -1).numpy()

        # PCA
        pca = PCA(n_components=self.pca_variance, svd_solver='full')
        weight_pca = pca.fit_transform(weight_matrix)
        weight_mean = pca.mean_
        components = pca.components_

        # Сохраняем всё в параметры / буферы
        self.register_buffer("weight_mean", torch.tensor(weight_mean, dtype=torch.float16, device=self.device))
        self.register_buffer("pca_components", torch.tensor(components, dtype=torch.float16, device=self.device))
        self.register_buffer("pca_weights", torch.tensor(weight_pca, dtype=torch.float16, device=self.device))

        # Сохраняем смещения
        if conv_layer.bias is not None:
            self.bias = nn.Parameter(conv_layer.bias.clone().detach().to(torch.float16), requires_grad=False)
        else:
            self.bias = None

    def restore_weights(self):
        # to float32 для избежания переполнений
        mean = self.weight_mean.to(torch.float16)
        components = self.pca_components.to(torch.float16)
        weights_pca = self.pca_weights.to(torch.float16)
        # quant_elements = mean.numel() + components.numel() + weights_pca.numel()
        # print("[INFO QuantizedConv2d] quant_elements: {}, source_elements: {}".format(quant_elements, self.source_elements))

        # (out_channels, n_components) @ (n_components, in_channels * kH * kW) + mean
        w = torch.matmul(weights_pca, components) + mean
        w = w.view(self.out_channels, self.in_channels, *self.kernel_size).to(torch.float16)
        return w

    def forward(self, x):
        weight = self.restore_weights()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class QuantizedConvTranspose2d(nn.Module):
    def __init__(self, conv_transpose_layer, pca_variance=Config.pca_accuracy):
        super().__init__()
        self.in_channels = conv_transpose_layer.in_channels
        self.out_channels = conv_transpose_layer.out_channels
        self.kernel_size = conv_transpose_layer.kernel_size
        self.stride = conv_transpose_layer.stride
        self.padding = conv_transpose_layer.padding
        self.output_padding = conv_transpose_layer.output_padding
        self.dilation = conv_transpose_layer.dilation
        self.groups = conv_transpose_layer.groups
        self.pca_variance = pca_variance
        self.device = conv_transpose_layer.weight.device
        self.source_elements = conv_transpose_layer.weight.numel()

        # Распакуем веса в матрицу: (out_channels, in_channels * kH * kW)
        w = conv_transpose_layer.weight.detach().cpu()
        weight_matrix = w.view(w.size(0), -1).numpy()
        # print("[INFO QuantizedConvTranspose2d] source weights: {}".format(w.numel()))
        # PCA
        pca = PCA(n_components=self.pca_variance, svd_solver='full')
        weight_pca = pca.fit_transform(weight_matrix)
        weight_mean = pca.mean_
        components = pca.components_

        # Сохраняем всё в параметры / буферы
        self.register_buffer("weight_mean", torch.tensor(weight_mean, dtype=torch.float16, device=self.device))
        self.register_buffer("pca_components", torch.tensor(components, dtype=torch.float16, device=self.device))
        self.register_buffer("pca_weights", torch.tensor(weight_pca, dtype=torch.float16, device=self.device))

        # Сохраняем смещения
        if conv_transpose_layer.bias is not None:
            self.bias = nn.Parameter(conv_transpose_layer.bias.clone().detach().to(torch.float16), requires_grad=False)
        else:
            self.bias = None

    def restore_weights(self):
        # to float32 для избежания переполнений
        mean = self.weight_mean.to(torch.float16)
        components = self.pca_components.to(torch.float16)
        weights_pca = self.pca_weights.to(torch.float16)
        # quant_elements = mean.numel() + components.numel() + weights_pca.numel()
        # print("[INFO QuantizedConvTranspose2d] quant_elements: {}, source_elements: {}".format(quant_elements, self.source_elements))

        # (out_channels, n_components) @ (n_components, in_channels * kH * kW) + mean
        w = torch.matmul(weights_pca, components) + mean
        w = w.view(self.in_channels, self.out_channels, *self.kernel_size).to(torch.float16)
        return w

    def forward(self, x):
        weight = self.restore_weights()
        return F.conv_transpose2d(x, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)