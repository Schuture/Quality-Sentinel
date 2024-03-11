import torch
from torch import nn, Tensor
import math
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class QualitySentinel(nn.Module):
    '''
    Quality Sentinel
    '''
    def __init__(self, hidden_dim=50, backbone='resnet50', embedding='text_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        if backbone == 'resnet18':
            self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if backbone == 'resnet34':
            self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if backbone == 'resnet50':
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if backbone == 'resnet101':
            self.net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        if backbone == 'resnet152':
            self.net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        flatten_feat_dim = 512 if backbone in ['resnet18', 'resnet34'] else 2048
        self.fc1 = nn.Linear(flatten_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.attention1 = nn.Linear(flatten_feat_dim+512, flatten_feat_dim)
        self.attention2 = nn.Linear(flatten_feat_dim+512, hidden_dim)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, embedding=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [N, flatten_feat_dim]
        
        att1 = self.attention1(torch.cat((x, embedding), dim=1))  # [N, flatten_feat_dim]
        att2 = self.attention2(torch.cat((x, embedding), dim=1))  # [N, hidden_dim]
        
        x = self.fc1(att1 * x)
        x = self.fc2(att2 * x)
        
        return x

