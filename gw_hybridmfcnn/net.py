import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
import torchvision.models as models


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
        'cnn': MFImageCNN,
        'cnn_small': MFImageCNN_Small,
        'resnet50': MFImageResnet50,
        'resnet18': MFImageResnet18,
        'vit_b_16': MFImageViTB16
    }
    if config.net.modelname in modeldict.keys():
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
            in_height, in_width = self._get_output_size(in_height, in_width, kernel_size, stride, padding, dilation)
            layers.append(get_activation_layer(config.net.activation))
            # Pooling layers
            kernel_size = config.net[f"pool_{i+1}_kernel_size"]
            stride = config.net[f"pool_{i+1}_stride"]
            padding = config.net[f"pool_{i+1}_padding"]
            dilation = config.net[f"pool_{i+1}_dilation"]
            layers.append(nn.MaxPool2d(kernel_size, stride, padding, dilation))
            in_height, in_width = self._get_output_size(in_height, in_width, kernel_size, stride, padding, dilation)
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        # Linear layers
        in_features = in_channels
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

    def _get_output_size(self, Hin: int, Win: int, kernel_size, stride, padding, dilation):
        if not isinstance(kernel_size, int):
            kh = kernel_size[0]
            kw = kernel_size[1]
        else:
            kh = kernel_size
            kw = kernel_size
        Hout = math.floor((Hin + 2 * padding - dilation * (kh - 1) - 1) / stride + 1)
        Wout = math.floor((Win + 2 * padding - dilation * (kw - 1) - 1) / stride + 1)
        return (Hout, Wout)


class MFImageCNN_Small(nn.Module):
    # CNN model for spectrogram input
    def __init__(self, config: DictConfig):
        super(MFImageCNN_Small, self).__init__()
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


class MFImageResnet18(nn.Module):
    def __init__(self, config: DictConfig):
        super(MFImageResnet18, self).__init__()
        self.net = models.resnet18(weights=None)

        # Chenge the channels of the first convolutional layer
        self.net.conv1 = nn.Conv2d(
            in_channels=config.net.input_channel,
            out_channels=self.net.conv1.out_channels,
            kernel_size=self.net.conv1.kernel_size,
            stride=self.net.conv1.stride,
            padding=self.net.conv1.padding,
            bias=self.net.conv1.bias
        )
        
        # Change the last layer
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, config.net.out_features)

    def forward(self, x):
        x = self.net(x)
        return x


class MFImageResnet50(nn.Module):
    def __init__(self, config: DictConfig):
        super(MFImageResnet50, self).__init__()
        self.net = models.resnet50(weights=None)

        # Chenge the channels of the first convolutional layer
        self.net.conv1 = nn.Conv2d(
            in_channels=config.net.input_channel,
            out_channels=self.net.conv1.out_channels,
            kernel_size=self.net.conv1.kernel_size,
            stride=self.net.conv1.stride,
            padding=self.net.conv1.padding,
            bias=self.net.conv1.bias
        )
        
        # Change the last layer
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, config.net.out_features)

    def forward(self, x):
        x = self.net(x)
        return x


class MFImageViTB16(nn.Module):
    def __init__(self, config: DictConfig):
        super(MFImageViTB16, self).__init__()
        assert config.net.input_height == config.net.input_width, "For vit_b_16, input height and input width must be equal."
        self.net = models.vit_b_16(weights=None, image_size=config.net.input_height)

        # Chenge the channels of the first convolutional layer
        self.net.conv_proj = nn.Conv2d(
            in_channels=config.net.input_channel,
            out_channels=self.net.conv_proj.out_channels,
            kernel_size=self.net.conv_proj.kernel_size,
            stride=self.net.conv_proj.stride,
            padding=self.net.conv_proj.padding
        )
        
        # Change the last layer
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, config.net.out_features)

    def forward(self, x):
        x = self.net(x)
        return x