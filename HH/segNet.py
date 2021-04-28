import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU())
        
        self.enconv1_1 = CBR(3, 64, 3, 1, 1)
        self.enconv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)
        
        self.enconv2_1 = CBR(64, 128, 3, 1, 1)
        self.enconv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)

        self.enconv3_1 = CBR(128, 256, 3, 1, 1)
        self.enconv3_2 = CBR(256, 256, 3, 1, 1)
        self.enconv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)
        
        self.enconv4_1 = CBR(256, 512, 3, 1, 1)
        self.enconv4_2 = CBR(512, 512, 3, 1, 1)
        self.enconv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)
        
        self.enconv5_1 = CBR(512, 512, 3, 1, 1)
        self.enconv5_2 = CBR(512, 512, 3, 1, 1)
        self.enconv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True, return_indices=True)
        
        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.deconv5_1 = CBR(512, 512, 3, 1, 1)
        self.deconv5_2 = CBR(512, 512, 3, 1, 1)
        self.deconv5_3 = CBR(512, 512, 3, 1, 1)
        
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.deconv4_1 = CBR(512, 512, 3, 1, 1)
        self.deconv4_2 = CBR(512, 512, 3, 1, 1)
        self.deconv4_3 = CBR(512, 256, 3, 1, 1)
        
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.deconv3_1 = CBR(256, 256, 3, 1, 1)
        self.deconv3_2 = CBR(256, 256, 3, 1, 1)
        self.deconv3_3 = CBR(256, 128, 3, 1, 1)
        
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.deconv2_1 = CBR(128, 128, 3, 1, 1)
        self.deconv2_2 = CBR(128, 64, 3, 1, 1)
        
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconv1_1 = CBR(64, 64, 3, 1, 1)
        
        self.score_fr = nn.Conv2d(64, num_classes, 3, 1, 1, 1)
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        
    def forward(self, x):
        h = self.enconv1_1(x)
        h = self.enconv1_2(h)
        h, pool1 = self.pool1(h)
        
        h = self.enconv2_1(h)
        h = self.enconv2_2(h)
        h, pool2 = self.pool2(h)
        
        h = self.enconv3_1(h)
        h = self.enconv3_2(h)
        h = self.enconv3_3(h)
        h, pool3 = self.pool3(h)
        
        h = self.enconv4_1(h)
        h = self.enconv4_2(h)
        h = self.enconv4_3(h)
        h, pool4 = self.pool4(h)
        
        h = self.enconv5_1(h)
        h = self.enconv5_2(h)
        h = self.enconv5_3(h)
        h, pool5 = self.pool5(h)
        
        h = self.unpool5(h, pool5)
        h = self.deconv5_1(h)
        h = self.deconv5_2(h)
        h = self.deconv5_3(h)
        
        h = self.unpool4(h, pool4)
        h = self.deconv4_1(h)
        h = self.deconv4_2(h)
        h = self.deconv4_3(h)
        
        h = self.unpool3(h, pool3)
        h = self.deconv3_1(h)
        h = self.deconv3_2(h)
        h = self.deconv3_3(h)
        
        h = self.unpool2(h, pool2)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)
        
        h = self.unpool1(h, pool1)
        h = self.deconv1_1(h)

        h = self.score_fr(h)
        
        return h