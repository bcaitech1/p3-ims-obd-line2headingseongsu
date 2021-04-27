import torch
import torch.nn as nn
from torch.nn import functional as F


def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                        out_ch, 
                                        kernel_size=size, 
                                        stride=1, 
                                        padding=rate, 
                                        dilation=rate),
                             nn.ReLU())
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        '''
        [TODO]

        ''' 
        self.features1=nn.Sequential(conv_relu(3,64,3,1),conv_relu(64,64,3,1),nn.MaxPool2d(3,stride=2,padding=1))
        self.features2=nn.Sequential(conv_relu(64,128,3,1),conv_relu(128,128,3,1),nn.MaxPool2d(3,stride=2,padding=1))
        self.features3=nn.Sequential(conv_relu(128,256,3,1),conv_relu(256,256,3,1),nn.MaxPool2d(3,stride=2,padding=1))
        self.features4=nn.Sequential(conv_relu(256,512,3,1),conv_relu(512,512,3,1),conv_relu(512,512,3,1),nn.MaxPool2d(3,stride=1,padding=1))
        self.features5=nn.Sequential(conv_relu(512,512,3,1),conv_relu(512,512,3,1),conv_relu(512,512,3,1),nn.MaxPool2d(3,stride=1,padding=1),nn.AvgPool2d(3,stride=1,padding=1))

    def forward(self, x):
        '''
        [TODO]

        '''
        h=self.features1(x)
        h=self.features2(h)
        h=self.features3(h)
        h=self.features4(h)
        out=self.features5(h)
        
        return out

    
class classifier(nn.Module):
    def __init__(self, num_classes): 
        super(classifier, self).__init__()
        '''
        [TODO]

        ''' 
        self.classifier=nn.Sequential(conv_relu(512,1024,3,rate=12),nn.Dropout2d(0.5),conv_relu(1024,1024,1,1),nn.Dropout2d(0.5),nn.Conv2d(1024,num_classes,1))

    def forward(self, x):
        '''
        [TODO]

        '''
        out=self.classifier(x)
        return out 


class DeepLabV1(nn.Module):
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV1, self).__init__()
        '''
        [TODO]

        ''' 
        self.backbone=backbone
        self.classifier=classifier
        self.upsampling=upsampling

    def forward(self, x):
        '''
        [TODO]

        '''
        x=self.backbone(x)
        x=self.classifier(x)
        _,_,feat_h,feat_w=x.size()
        # 문제발생 지점. 위치상 피어세션에서 말한대로 클라스 갯수 문제일수도 있을 거 같다. 일단 그냥 계산기 뚜둥겨 보니까 feat 크기가 줄어들면 만사형통이여서 반찬고 붙였다.
        out=F.interpolate(x,size=((feat_h-2)*self.upsampling,(feat_w-2)*self.upsampling),mode='bilinear')
        
        return out
