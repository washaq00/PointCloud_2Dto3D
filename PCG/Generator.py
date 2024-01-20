import torch.nn as nn
from PCG.Encoder.ResNet import ResNet50
import torch
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_points=1024):
        super(Generator, self).__init__()
        self.resnet = ResNet50(pretrained=True)
        self.num_points = num_points
        self.zmean = nn.Linear(1000, 100)
        self.zlog = nn.Linear(1000, 100)
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        zmean = self.zmean(x)
        zlog = self.zlog(x)
        zsigma = torch.sqrt(torch.exp(zlog))
        eps = torch.randn(zmean.size()).cuda()
        z = zmean + zsigma * eps

        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = self.th(self.fc3(x))
        print(x.size())
        p = x.view(batchsize, 3, self.num_points)
        print(p.size())
        return p, zmean, zsigma, z