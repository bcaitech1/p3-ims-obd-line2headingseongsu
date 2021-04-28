# 모델 참고 코드 
# https://github.com/wkentaro/pytorch-fcn/
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import vgg16

class FCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__()
        self.pretrained_model = vgg16(pretrained=True)
        features, classfiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())
        
        self.feature_map1 = nn.Sequential(*features[0:24])
        self.feature_map2 = nn.Sequential(*features[24:])
        
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)
        
        self.conv_67 = nn.Sequential(nn.Conv2d(512,4096,1),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),
                                     nn.Conv2d(4096,4096,1),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout())
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        
        self.upscore1 = nn.ConvTranspose2d(num_classes, num_classes,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes,
                                          kernel_size=32,
                                          stride=16,
                                          padding=8)
        

    def forward(self, x):
        pool4 = h = self.feature_map1(x)
        
        h = self.feature_map2(h)
        h = self.conv_67(h)
        h = self.score_fr(h)
        h = self.upscore1(h)
        
        score_pool4 = self.score_pool4_fr(pool4)
        
        output = score_pool4 + h
        
        output = self.upscore2(output)
        
        return output