import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from networks.deeplabv3plus import DeepLabV3Plus, apply_channel_mask


def test_shapes_and_mask():
    torch.manual_seed(0)
    net = DeepLabV3Plus(pretrained=False).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        z, low = net.encode(x)
        assert z.shape == (2, 2048, 14, 14) and low.shape == (2, 256, 56, 56)
        out = net(x)
        assert out.shape == (2, 3, 224, 224)
        assert torch.equal(out, net(x, mask_channels=[]))
        assert torch.equal(out, net.decode(z, low, (224, 224)))
        zm = apply_channel_mask(z, [0, 5])
        assert zm[:, [0, 5]].abs().sum() == 0
        keep = [c for c in range(2048) if c not in (0, 5)]
        assert torch.equal(zm[:, keep], z[:, keep])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
