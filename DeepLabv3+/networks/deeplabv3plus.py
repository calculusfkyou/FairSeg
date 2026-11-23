"""DeepLabV3+ with a ResNet-50 encoder (spec §7), split into encode / decode so the layer4 map Z can be masked."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def apply_channel_mask(z, channels):
    """Zero the given channels of z at every position; None or empty returns z unchanged."""
    if not channels:
        return z
    z = z.clone()
    z[:, list(channels)] = 0
    return z


def _conv_bn_relu(cin, cout, k, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, cin=2048, cout=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([_conv_bn_relu(cin, cout, 1)] + [_conv_bn_relu(cin, cout, 3, r) for r in rates])
        # No BN on the pooled branch, as in experiment_001: it is 1×1 per image.
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(cout * (len(rates) + 2), cout, 1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([b(x) for b in self.branches] + [pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                                 replace_stride_with_dilation=(False, False, True))
        self.backbone.avgpool, self.backbone.fc = nn.Identity(), nn.Identity()
        self.aspp = ASPP()
        self.low_proj = _conv_bn_relu(256, 48, 1)
        self.decoder = nn.Sequential(_conv_bn_relu(256 + 48, 256, 3), _conv_bn_relu(256, 256, 3),
                                     nn.Conv2d(256, num_classes, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        for head in (self.aspp, self.low_proj, self.decoder):
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        """x in [0, 1] -> (Z = layer4 output, low = layer1 output)."""
        b = self.backbone
        x = (x - self.mean) / self.std
        low = b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x)))))
        return b.layer4(b.layer3(b.layer2(low))), low

    def decode(self, z, low, size):
        y = F.interpolate(self.aspp(z), size=low.shape[-2:], mode='bilinear', align_corners=False)
        y = self.decoder(torch.cat([y, self.low_proj(low)], dim=1))
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False)

    def forward(self, x, mask_channels=None):
        z, low = self.encode(x)
        return self.decode(apply_channel_mask(z, mask_channels), low, x.shape[-2:])
