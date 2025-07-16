import math
import torch
import torch.nn as nn
from omegaconf import DictConfig


class get_activation_layer(nn.Module):
    """
    PyTorch module representing an activation function layer.
    """
    def __init__(self, activation_fn: str):
        super(get_activation_layer, self).__init__()
        self.activation_fn_name = activation_fn
        self.activation_fn = getattr(nn.modules.activation, activation_fn)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(x)


def instantiate_neuralnetwork(config: DictConfig):
    modeldict = {
        'cnn': MFImageCNN
    }
    if config.net.modelname in ['cnn']:
        return modeldict[config.net.modelname](config)
    else:
        net = modeldict[config.net.modelname](weights=config.net.model_weight)
        for param in net.parameters():
            param.requires_grad = False
        net.heads[0] = nn.Linear(net.heads[0].in_features, config.net.out_features)
        return net


class MFImageCNN(nn.Module):
    # CNN model for spectrogram input
    def __init__(self, config: DictConfig):
        super(MFImageCNN, self).__init__()
        layers = []
        in_channels = config.net.input_channel
        in_height = config.net.input_height
        in_width = config.net.input_width
        for i in range(config.net.num_conv_layers):
            # Convolutional layers
            out_channels = config.net[f"conv_{i+1}_out_channels"]
            kernel_size = config.net[f"conv_{i+1}_kernel_size"]
            stride = config.net[f"conv_{i+1}_stride"]
            padding = config.net[f"conv_{i+1}_padding"]
            dilation = 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
            in_height = math.floor((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            in_width = math.floor((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            layers.append(get_activation_layer(config.net.activation))
            # Pooling layers
            kernel_size = config.net[f"pool_{i+1}_kernel_size"]
            stride = config.net[f"pool_{i+1}_stride"]
            padding = config.net[f"pool_{i+1}_padding"]
            dilation = config.net[f"pool_{i+1}_dilation"]
            layers.append(nn.MaxPool2d(kernel_size, stride, padding, dilation))
            in_height = math.floor((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            in_width = math.floor((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            in_channels = out_channels
        layers.append(nn.Flatten())
        # Linear layers
        in_features = in_channels * in_height * in_width
        for i in range(config.net.num_linear_layers):
            out_features = config.net[f"linear_{i+1}_size"]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(get_activation_layer(config.net.activation))
            in_features = out_features
        layers.append(nn.Linear(in_features, config.net.out_features))
        # Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
