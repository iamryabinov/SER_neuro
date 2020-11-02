import os
from torchsummary import summary

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
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




class AlexNetMultiTask(nn.Module):

    def __init__(self, num_emotions=4, num_speakers=10, num_genders=2):
        super(AlexNetMultiTask, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.joint_classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.classifier_emotion = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_emotions)
        )
        self.classifier_speaker = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_speakers)
        )
        self.classifier_gender = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_genders)
        )

    def forward(self, x):
        hidden = self.features(x)
        hidden = self.avgpool(hidden)
        hidden = hidden.flatten(1)
        hidden = self.joint_classifier(hidden)
        emotion_prediction = self.classifier_emotion(hidden)
        speaker_prediction = self.classifier_speaker(hidden)
        gender_prediction = self.classifier_gender(hidden)
        return emotion_prediction, speaker_prediction, gender_prediction


class VGG(nn.Module):

    def __init__(self, features, num_emotions=4, num_speakers=10, num_genders=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.joint_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.classifier_emotion = nn.Sequential(
            nn.Linear(2048, 512),
            # nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_emotions)
        )
        self.classifier_speaker = nn.Sequential(
            nn.Linear(2048, 512),
            # nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_speakers)
        )
        self.classifier_gender = nn.Sequential(
            nn.Linear(2048, 512),
            # nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_genders)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        hidden = self.features(x)
        hidden = self.avgpool(hidden)
        hidden = hidden.flatten(1)
        hidden = self.joint_classifier(hidden)
        emotion_prediction = self.classifier_emotion(hidden)
        speaker_prediction = self.classifier_speaker(hidden)
        gender_prediction = self.classifier_gender(hidden)
        return emotion_prediction, speaker_prediction, gender_prediction

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)

def vgg(num_emotions, num_speakers, num_genders, type=16, bn=False):
    choice_dict = {
        (11, False): vgg11(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (11, True): vgg11_bn(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (13, False): vgg13(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (13, True): vgg13_bn(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (16, False): vgg16(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (16, True): vgg16_bn(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (19, False): vgg19(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
        (19, True): vgg19_bn(num_emotions=num_emotions, num_speakers=num_speakers, num_genders=num_genders),
    }
    choice = (type, bn)
    return(choice_dict[choice])

if __name__ == '__main__':
    pass