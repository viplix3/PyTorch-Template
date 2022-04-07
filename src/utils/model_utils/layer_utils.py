import torch
import torch.nn as nn

from typing import List
BN_MOMENTUM = 0.1

class ConvBatchNormRelu(nn.Module):
    def __init__(self, nIn: int, nOut: int, kSize: int, stride:int = 1, padding: int = None):
        """ This class defines a convolution layers with BatchNorm and ReLU activation

        Args:
            nIn (int): Number of input channels
            nOut (int): Number of output channels
            kSize (int): Convolution kernel size
            stride (int, optional): Stride for applying convolution operation. Defaults to 1.
            padding (int, optional): Padding to be used when applying convolution. Defaults to None.
        """
        super(ConvBatchNormRelu, self).__init__()
        padding = padding if padding is not None else int((kSize) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
                    padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=BN_MOMENTUM)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input feature map

        Returns:
            torch.Tensor: convolved feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)

        return output


class ConvBatchNorm(nn.Module):
    def __init__(self, nIn: int, nOut: int, kSize: int, stride: int = 1, padding: int = None):
        """ This class defines a convolution layers with BatchNorm

        Args:
            nIn (int): Number of input channels
            nOut (int): Number of output channels
            kSize (int): Convolution kernel size
            stride (int, optional): Stride for applying convolution operation. Defaults to 1.
            padding (int, optional): Padding to be used when applying convolution. Defaults to None.
        """
        super(ConvBatchNorm, self).__init__()
        padding = padding if padding is not None else int((kSize) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
                    padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=BN_MOMENTUM)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input feature map

        Returns:
            torch.Tensor: convolved feature map
        """
        output = self.conv(input)
        output = self.bn(output)

        return output


class Downsampler(nn.Module):
    def __init__(self, nIn: int, nOut: int):
        """ This class defines a max-pool and convolution hybrid 2x downsampler

        Args:
            nIn (int): Number of input channels
            nOut (int): Number of output channels
        """
        super(Downsampler, self).__init__()

        self.max_pool_downsampling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_downsampling = ConvBatchNorm(nIn, nIn, kSize=3, stride=2, padding=1)

        misc_layers: List[nn.Module] = []
        if(nIn != nOut):
            if(nOut == (2*nIn)):
                misc_layers.append(nn.ReLU(inplace=True))
            else:
                misc_layers.append(ConvBatchNormRelu(nIn*2, nOut, kSize=1))
        else:
            misc_layers.append(ConvBatchNormRelu(nIn*2, nIn, kSize=1))
        self.misc_layers = nn.Sequential(*misc_layers)

    def apply_concat(self, outs: List[torch.Tensor]) -> torch.Tensor:
        """ Concatenates the input list of tensors along the batch dimension

        Args:
            outs (List[torch.Tensor]): list of torch.Tensor for applying concatenation

        Returns:
            torch.Tensor: Concatenated output
        """
        ret_outs = torch.cat(outs, dim=1)
        return ret_outs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input feature map

        Returns:
            torch.Tensor: downsampled feature map
        """
        max_pool_out = self.max_pool_downsampling(input)
        conv_pool_out = self.conv_downsampling(input)
        downsampled_output = self.apply_concat([max_pool_out, conv_pool_out])
        downsampled_output = self.misc_layers(downsampled_output)
        return downsampled_output