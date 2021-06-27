import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ReLU
from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import torch.nn.modules.conv
# from conv import Conv

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def parse(in_planes: int, out_planes: int,  stride: int = 1) -> nn.Conv2d:
    return Parse(in_planes, out_planes, stride=stride)

def conv(in_planes: int, out_planes: int, kernel_size=1, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1*1 convolution without padding"""
    return Conv(in_planes, out_planes, kernel_size, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# def parse(l, ch_out, stride, channel_wise = True):
#     shape = l.get_shape().as_list()
#     l_ori = l
#     parse1 = nn.Sigmoid(nn.Conv('parse1', l, 1, 1, strides=1))
#     l1_left = l * parse1
#     l = l - l1_left

#     parse2 = tf.nn.sigmoid(Conv('parse2', l, 1, 1, strides=1))
#     l2_left = l * parse2
#     l = l - l2_left

#     parse3 = tf.nn.sigmoid(Conv('parse3', l, 1, 1, strides=1))
#     l3_left = l * parse3
#     l3_right = l - l3_left

#     l1_left = tf.keras.backend.resize_images(Conv2D('l1_left',  AvgPooling('pool', l1_left, pool_size=8, strides=8, padding='VALID'),
#              1*ch_out//4,  3 if shape[1]//8 > 2  else 1, strides=1), 8//stride, 8//stride, 'channels_last' )
#     l2_left = tf.keras.backend.resize_images(Conv2D('l2_left', AvgPooling('pool', l2_left, pool_size=4, strides=4, padding='VALID'),
#              1*ch_out//4, 3 if shape[1]//4 > 2  else 1, strides=1), 4//stride, 4//stride, 'channels_last')
#     l3_left = tf.keras.backend.resize_images(Conv2D('l3_left',  AvgPooling('pool', l3_left, pool_size=2, strides=2, padding='VALID'),
#              1*ch_out//4, 3 if shape[1]//2 > 2 else 1, strides=1), 2//stride, 2//stride, 'channels_last')
#     l3_right = Conv2D('l3_right',  l3_right,
#              1*ch_out//4, 3 if shape[1] > 2 else 1, strides=stride)

#     l_ori = Conv2D('l_ori', l_ori, ch_out//4, 3, strides=stride, activation = BNReLU)

#     l = tf.concat([tf.nn.sigmoid(BatchNorm('bn1', l1_left))*l_ori, tf.nn.sigmoid(BatchNorm('bn2',l2_left))*l_ori,
#           tf.nn.sigmoid(BatchNorm('bn3', l3_left))*l_ori, tf.nn.sigmoid(BatchNorm('bn4', l3_right))*l_ori], -1)
#     return l
#tensorpack版本的修改为树卷积的地方
# def resnet_bottleneck(l, ch_out, stride, stride_first=False):
#     """
#     stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
#     """
#     shortcut = l
#     l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
#     l = parse(l, ch_out, stride = stride)
#     l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
#     out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
#     return tf.nn.relu(out)



#修改版
#self.conv2 = parse(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
class Parse(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Parse, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1 在跨度！= 1时，self.conv2和self.downsample层都会对输入进行下采样

        self.shape = [4, 64, 56, 56]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #定义parse的各层

        self.sigmoid = nn.Sigmoid()
        # self.conv1 = conv(in_channels, 1, 1, stride=1)
        # self.conv2 = conv(in_channels, 1, 1, stride=1)
        # self.conv3 = conv(in_channels, 1, 1, stride=1)
        self.conv1 = conv1x1(in_channels, 1, stride=1)
        self.conv2 = conv1x1(in_channels, 1, stride=1)
        self.conv3 = conv1x1(in_channels, 1, stride=1)


        self.Relu = ReLU()

        self.avgpooling1 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # tensorpack和pytorch的shape有区别,而且pytorch的3核卷积和二核卷积的参数写法不同（padding=1）所以写了以下两个，在forward函数里面定义调用哪个
        self.conv2D1_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D1_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        self.avgpooling2 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        # self.conv2D2 = nn.Conv2d(in_channels,1*self.out_channels//4, 3 if self.shape[1]//4 > 2  else 1, stride=1)
        self.conv2D2_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D2_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        self.avgpooling3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2D3 = nn.Conv2d(in_channels,1*self.out_channels//4, 3 if self.shape[1]//2 > 2 else 1, stride=1)
        self.conv2D3_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D3_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        # self.conv2D4 = nn.Conv2d(in_channels,1*self.out_channels//4, 3 if self.shape[1] > 2 else 1, stride=stride)
        self.conv2D4_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=stride)
        self.conv2D4_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=stride)

        self.conv2D5 = conv3x3(in_channels, 1*self.out_channels//4, stride=stride)

        # self.bn1 = norm_layer(self.out_channels//4)
        self.bn1 = norm_layer(self.in_channels//4)
        self.bn2 = norm_layer(self.in_channels//4)
        self.bn3 = norm_layer(self.in_channels//4)
        self.bn4 = norm_layer(self.in_channels//4)

    #前向传播
    def forward(self, l: Tensor) -> Tensor:
        l_ori = l

        l_ori = self.conv2D5(l_ori)
        l_ori = self.Relu(l_ori)

        parse1 = self.conv1(l)
        parse1 = self.sigmoid(parse1)
        l1_left = l * parse1
        l = l - l1_left

        parse2 = self.conv2(l)
        parse2 = self.sigmoid(parse2)
        l2_left = l * parse2
        l = l - l2_left

        parse3 = self.conv3(l)
        parse3 = self.sigmoid(parse3)
        l3_left = l * parse3
        l3_right = l - l3_left

        out_avgpooling1 = self.avgpooling1(l1_left)
        if(self.shape[2]//8 > 2):
            out_conv2D1 = self.conv2D1_3(out_avgpooling1)
        else:
            out_conv2D1 = self.conv2D1_1(out_avgpooling1)
        # l1_left = tf.keras.backend.resize_images(out_conv2D1, 8//self.stride, 8//self.stride, 'channels_last' )
        rs = Tuple[int,...]
        # rs = [out_conv2D1.shape[2]*8//self.stride, out_conv2D1.shape[3]*8//self.stride]
        rs = [l_ori.shape[2],l_ori.shape[3]]
        l1_left = nn.functional.interpolate(out_conv2D1, size = rs, mode = 'nearest')

        out_avgpooling2 = self.avgpooling2(l2_left)
        # tensorpack和pytorch的shape有区别
        if(self.shape[2]//4 > 2):
            out_conv2D2 = self.conv2D2_3(out_avgpooling2)
        else:
            out_conv2D2 = self.conv2D2_1(out_avgpooling2)
        # l2_left = tf.keras.backend.resize_images(out_conv2D2, 4//self.stride, 4//self.stride, 'channels_last')
        ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #找到了原因：l2是7 *4  l1是3*8!!!  原因是7Apooling之后，得到3.5，但是Pool是直接取整
        
        l2_left = nn.functional.interpolate(out_conv2D2, size = rs, mode = 'nearest')

        out_avgpooling3 = self.avgpooling3(l3_left)
        if(self.shape[2]//2 > 2):
            out_conv2D3 = self.conv2D3_3(out_avgpooling3)
        else:
            out_conv2D3 = self.conv2D3_1(out_avgpooling3)
        # l3_left = tf.keras.backend.resize_images(out_conv2D3, 2//self.stride, 2//self.stride, 'channels_last')
        l3_left = nn.functional.interpolate(out_conv2D3, size = rs, mode = 'nearest')

        if(self.shape[2] > 2):
            l3_right = self.conv2D4_3(l3_right)
        else:
            l3_right = self.conv2D4_1(l3_right)

        bn1 = self.bn1(l1_left)
        bn2 = self.bn2(l2_left)
        bn3 = self.bn3(l3_left)
        bn4 = self.bn4(l3_right)

        l = torch.cat([self.sigmoid(bn1) * l_ori, self.sigmoid(bn2) * l_ori,
            self.sigmoid(bn3) * 0.5, self.sigmoid(bn4) * 0.5], 1)
        return l




class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.Relu = ReLU()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1 在跨度！= 1时，self.conv2和self.downsample层都会对输入进行下采样
        #定义bottleneck的各层

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # print('bn1', width)
        
        # def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
            # """3x3 convolution with padding"""
            # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            #                  padding=dilation, groups=groups, bias=False, dilation=dilation)

        # #tensorpackori
        # l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
        # # tensorpack treeconv
        # l = parse(l, ch_out, stride = stride)
        #print('width, planes', width,planes)

        #pytorch ｐａｒｓｅ
        self.conv2 = parse(width, planes, stride=stride)
        # pytorch ori:
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        
        self.bn2 = norm_layer(width)


        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    #前向传播
    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        
        out = self.conv2(out)
       


        out = self.bn2(out)
        
        out = self.conv3(out)
        
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
       
        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #!!!!!在这里下面变换图片大小
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])#512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print(x.size())#torch.Size([32, 3, 224, 224])   # torch.Size([25, 3, 84, 84]) torch.Size([50, 3, 84, 84])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)

        ###parse操作，修改大小
        shape = x.shape
        # print("x1:",x.shape)
        """x1: torch.Size([25, 256, 21, 21])"""
        # l = tf.image.resize_images(l, [shape[1]//7*8, shape[2]//7*8], method=0)
        x = nn.functional.interpolate(x, [shape[2]//7*8, shape[3]//7*8], mode = 'nearest')
        
        # x: torch.Size([25, 256, 24, 24])"""
        x = self.layer2(x)
        # # print(x.size())# ori: torch.Size([32, 512, 28, 28])  parse:torch.Size([32, 512, 32, 32])
        # x = self.layer3(x)
        # # print(x.size())# ori: torch.Size([32, 1024, 14, 14])  parse:torch.Size([32, 1024, 16, 16])
        # x = self.layer4(x)
        # print(x.size())# ori:torch.Size([32, 2048, 7, 7])  parse:torch.Size([32, 2048, 8, 8])
        x = self.avgpool(x)
        # # print(x.size())#ori 和　parse : torch.Size([32, 2048, 1, 1])
        x = torch.flatten(x, 1)
        # print('x = torch.flatten(x, 1)', x.size())#ori 和　parse : torch.Size([32, 2048])
        # x = self.fc(x)
        # print(x.size())#　ori 和　parse : torch.Size([32, 1000])

        #ＲＮ调整图形大小
        x = x.view(x.shape[0],8,8,8)#resnet50 裁剪版
        # x = x.view(x.shape[0],32,8,8)#resnet50
        # x = x.view(x.shape[0],32,4,4)#resnet18

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# def resnet_bottleneck_ori(l, ch_out, stride, stride_first=False):
#     """
#     stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
#     """
#     shortcut = l
#     l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
#     l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
#     l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
#     out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
#     return tf.nn.relu(out)
