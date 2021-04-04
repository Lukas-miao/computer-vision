import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url  # 引入pytorch中已经预训练的权重
try:
    from torch.hub import load_state_dict_from_url
    # pytorch hub 是一个预训练模型库（包括模型定义和预训练权重）
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
# from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Union, List, Dict, Any, cast  # 表达复杂的类型表达情况

_all_ = [
    'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]
model_urls = {
    'vgg11': 'http://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'http://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'http://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'http://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'http://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'http://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'http://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'http://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# nn.module为VGG的父类
class VGG(nn.Module):
    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 实现全连接层
        self.classifier = nn.Sequential(
            nn.linear(512 * 7 * 7, 4096),  # 将特征矩阵展平，元素个数为512*7*7，输出节点个数为4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 随机失活，以50%的概率
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )   # 分类层softmax layer
        if init_weights:
            self._initialize_weights()  # 如果输入的init_weights为true时，则进入初始化权重函数中

    # 正向传播的过程
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # torch.Tensor用于生成新的张量
        x = self.features(x)       # 特征网络的结构
        x = self.avgpool(x)
        x = torch.flatten(x, 1)    # 展平操作，第0个维度是batch维度，所以从第一个维度开始展平
        x = self.classifier(x)     # 展平后输入全连接层进行分类
        return x

    # 初始化权重函数，遍历网络的每一个子模块即每一层，若为卷积层，则使用一种初始化方法来初始化卷积核的权重，kaiming或者xavier
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bis is not None:                # 若卷积核采用了偏置，则将偏置默认初始化为零
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # 批次归一化的判断
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.linear):       # 全连接层的判断
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 传入配置变量，生成特征提取网络结构
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers = []  # : List[nn.Module]
    in_channels = 3  # 彩色图片
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)  # 数据类型转换
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # stride默认为1，可略
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # 批次规范化（也属于正则方式的的一种），将数据归一化，提高训练效率
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 特征图通过该层卷积后，输出的深度就变成卷积核的个数，即v
        return nn.Sequential(*layers)
        # (*layers):形参——单个星号代表这个位置接收任意多个非关键字参数，转化成元组方式。实参——如果*号加在了是实参上，代表的是将输入迭代器拆成一个个元素。


# 字典，表示四种网络配置，参照VGG网络配置表(全部使用3*3卷积，字典表明卷积核个数和池化层，使用最大池化)
cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


# 实例化vgg网络
def _vgg(arch: str, cfg: str, batch: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm = batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url[arch], progress=progress)
        model.load_state_dict(state_dict)
    print(model)


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layers model (configuration "A") from
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layers model (configuration "A") with batch normalization
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layers model (configuration "B") from
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layers model (configuration "B") with batch normalization
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layers model (configuration "D") from
     "ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layers model (configuration "D") with batch normalization
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layers model (configuration "E") from
     "ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layers model (configuration "E") with batch normalization
     `"ver deep convolutional networks for large-scale image recognition"<https://arxiv.org/pdf/1409.1556.pdf>`._
      Args:
          pretrained(bool): if true,returns a model pre-trained on ImageNet
          progress(bool): if true,displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
