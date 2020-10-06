import torch
from torchvision import models


# modify VGG16 to leave out the last layer
class vgg16mod(torch.nn.Module):
    def __init__(self):
        super(vgg16mod, self).__init__()
        vgg16 = models.vgg16(pretrained=True, progress=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = torch.nn.Sequential(*list(vgg16.classifier)[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class resnetmod(torch.nn.Module):
    def __init__(self):
        super(resnetmod, self).__init__()
        resnet = models.resnet18(pretrained=True, progress=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # skip the fully connected layer here!
        # x = self.fc(x)

        return x


class alexnetmod(torch.nn.Module):
    def __init__(self):
        super(alexnetmod, self).__init__()
        alexnet = models.alexnet(pretrained=True, progress=True)

        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = torch.nn.Sequential(*list(alexnet.classifier)[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
