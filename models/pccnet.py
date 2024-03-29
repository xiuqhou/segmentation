from torchvision.models import resnet50
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.register_buffer('inf', torch.tensor(float("inf")))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def INF(self, B, H, W):
        return -torch.diag(self.inf.repeat(H), 0).unsqueeze(0).repeat(B*W, 1, 1)
    
    def forward(self, x):
        b, _, h, w = x.shape
        query = self.query_conv(x)
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h).permute(0, 2, 1)
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w).permute(0, 2, 1)

        key = self.key_conv(x)
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h)
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w)

        value = self.value_conv(x)
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h)
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w)

        energy_H = (torch.bmm(query_H, key_H)+self.INF(b,h,w)).view(b,w,h,h).permute(0, 2, 1, 3)
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,:h].permute(0, 2, 1, 3).contiguous().view(b*w,h,h)
        att_W = concate[:,:,:,h:h+w].contiguous().view(b*h,w,w)
        out_H = torch.bmm(value_H, att_H.permute(0, 2, 1)).view(b,w,-1,h).permute(0,2,3,1)
        out_W = torch.bmm(value_W, att_W.permute(0, 2, 1)).view(b,h,-1,w).permute(0,2,1,3)

        return self.gamma*(out_H + out_W) + x


class RCCAModule(nn.Module):
    """
    RCCAModule就是多个CrissCrossAttention的堆叠。

    Note:
        原作者out_channels=in_channels // 4，和inter_channels一致，这里直接用inter_channels代替。
    """
    def __init__(self, recurrence=2, in_channels=2048, num_classes=33):
        super().__init__()
        self.recurrence = recurrence
        inter_channels = in_channels // 4 
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.CCA = CrissCrossAttention(inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(inter_channels, num_classes, 1)
        )
    
    def forward(self, x):
        # reduce channels from C to C' 2048 -> 512
        output = self.conv_in(x)
        for i in range(self.recurrence):
            output = self.CCA(output)
        
        output = self.conv_out(output)
        output = self.cls_seg(torch.cat([x, output], 1))
        return output

class PCCNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # set backbone and classifier
        resnet_50 = resnet50(pretrained=True) # Construct a custom resnet50 based on resnet50 of pytorch
        self.backbone = nn.Module()
        self.backbone.stem = nn.Sequential(*list(resnet_50.children())[:4])
        for i in range(1, 5):
            setattr(self.backbone, f'layer{i}', getattr(resnet_50, f'layer{i}'))
            setattr(self, f'classifier{i}', RCCAModule(recurrence=2, in_channels=128*2**i, num_classes=num_classes))
        # set loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)


    def forward(self, x, label=None):
        input_shape = x.shape[-2:]
        # get model predictions from 4 stages
        x = self.backbone.stem(x)
        # NOTE: Here we should evaluate which of average outs and the last out can get better performance.
        loss, out = 0, 0
        weights = [0.05, 0.1, 0.4, 1]
        for i in range(1, 5):
            x = getattr(self.backbone, f'layer{i}')(x)
            y = getattr(self, f'classifier{i}')(x)
            y = F.interpolate(y, size=input_shape, mode='bilinear', align_corners=False)
            if self.training:
                loss += weights[i-1] * self.criterion(y, label) # deep supervision
            # out = out * (i-1) / i + y / i # calculate the average prediction
        if self.training:
            return loss
        return y

if __name__ == '__main__':
    model = PCCNet(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    model.cuda()
    out = model(x.cuda(), label=torch.zeros(2, 224, 224).cuda().long())
    print(out.shape)