import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ReLU
from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import torch.nn.modules.conv
# from conv import Conv



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def parse(in_planes: int, out_planes: int,  stride: int = 1) -> nn.Conv2d:
    return Parse(in_planes, out_planes, stride=stride)


#修改版
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #定义parse的各层

        self.sigmoid = nn.Sigmoid()
        self.conv1 = conv1x1(in_channels, 1, stride=1)
        self.conv2 = conv1x1(in_channels, 1, stride=1)
        self.conv3 = conv1x1(in_channels, 1, stride=1)

        self.Relu = ReLU()

        self.avgpooling1 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # tensorpack和pytorch的shape有区别,而且pytorch的3核卷积和二核卷积的参数写法不同（padding=1）所以写了以下两个，在forward函数里面定义调用哪个
        self.conv2D1_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D1_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        self.avgpooling2 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.conv2D2_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D2_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        self.avgpooling3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.conv2D3_3 = conv3x3(in_channels, 1*self.out_channels//4, stride=1)
        self.conv2D3_1 = conv1x1(in_channels, 1*self.out_channels//4, stride=1)

        self.conv2D5 = conv3x3(in_channels, 1*self.out_channels//4, stride=stride)

        self.bn1 = norm_layer(self.in_channels//4)
        self.bn2 = norm_layer(self.in_channels//4)
        self.bn3 = norm_layer(self.in_channels//4)

    #前向传播
    def forward(self, l: Tensor) -> Tensor:
        self.shape = l.shape
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
        l2_right = l - l2_left

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

        if(self.shape[2] > 2):
            l2_right = self.conv2D3_3(l2_right)
        else:
            l2_right = self.conv2D3_1(l2_right)

        bn1 = self.bn1(l1_left)
        bn2 = self.bn2(l2_left)
        bn3 = self.bn3(l2_right)

        l = torch.cat([self.sigmoid(bn1) * l_ori, self.sigmoid(bn2) * l_ori,
            self.sigmoid(bn3) * l_ori, self.sigmoid(bn3) * 0], 1)
        return l


    # #前向传播
    # def forward(self, l: Tensor) -> Tensor:
    #     # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',l.shape)
    #     self.shape = l.shape
    #     l_ori = l

    #     l_ori = self.conv2D5(l_ori)
    #     l_ori = self.Relu(l_ori)

    #     parse1 = self.conv1(l)
    #     parse1 = self.sigmoid(parse1)
    #     l1_left = l * parse1
    #     l = l - l1_left

    #     parse2 = self.conv2(l)
    #     parse2 = self.sigmoid(parse2)
    #     l2_left = l * parse2
    #     l = l - l2_left

    #     parse3 = self.conv3(l)
    #     parse3 = self.sigmoid(parse3)
    #     l3_left = l * parse3
    #     l3_right = l - l3_left

    #     out_avgpooling1 = self.avgpooling1(l1_left)
    #     if(self.shape[2]//8 > 2):
    #         out_conv2D1 = self.conv2D1_3(out_avgpooling1)
    #     else:
    #         out_conv2D1 = self.conv2D1_1(out_avgpooling1)
    #     # l1_left = tf.keras.backend.resize_images(out_conv2D1, 8//self.stride, 8//self.stride, 'channels_last' )
    #     rs = Tuple[int,...]
    #     # rs = [out_conv2D1.shape[2]*8//self.stride, out_conv2D1.shape[3]*8//self.stride]
    #     rs = [l_ori.shape[2],l_ori.shape[3]]
    #     l1_left = nn.functional.interpolate(out_conv2D1, size = rs, mode = 'nearest')

    #     out_avgpooling2 = self.avgpooling2(l2_left)
    #     # tensorpack和pytorch的shape有区别
    #     if(self.shape[2]//4 > 2):
    #         out_conv2D2 = self.conv2D2_3(out_avgpooling2)
    #     else:
    #         out_conv2D2 = self.conv2D2_1(out_avgpooling2)
    #     # l2_left = tf.keras.backend.resize_images(out_conv2D2, 4//self.stride, 4//self.stride, 'channels_last')
    #     ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     #找到了原因：l2是7 *4  l1是3*8!!!  原因是7Apooling之后，得到3.5，但是Pool是直接取整
        
    #     l2_left = nn.functional.interpolate(out_conv2D2, size = rs, mode = 'nearest')

    #     out_avgpooling3 = self.avgpooling3(l3_left)
    #     if(self.shape[2]//2 > 2):
    #         out_conv2D3 = self.conv2D3_3(out_avgpooling3)
    #     else:
    #         out_conv2D3 = self.conv2D3_1(out_avgpooling3)
    #     # l3_left = tf.keras.backend.resize_images(out_conv2D3, 2//self.stride, 2//self.stride, 'channels_last')
    #     l3_left = nn.functional.interpolate(out_conv2D3, size = rs, mode = 'nearest')

    #     if(self.shape[2] > 2):
    #         l3_right = self.conv2D4_3(l3_right)
    #     else:
    #         l3_right = self.conv2D4_1(l3_right)

    #     bn1 = self.bn1(l1_left)
    #     bn2 = self.bn2(l2_left)
    #     bn3 = self.bn3(l3_left)
    #     bn4 = self.bn4(l3_right)

    #     l = torch.cat([self.sigmoid(bn1) * l_ori, self.sigmoid(bn2) * l_ori,
    #         self.sigmoid(bn3) * l_ori, self.sigmoid(bn4) * l_ori], 1)
    #     return l



