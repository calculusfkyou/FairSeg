import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from utils import CEDiceLoss


def test_dice_not_squared_and_no_background():
    logits = torch.zeros(1, 3, 2, 2)                      # softmax 後每類都是 1/3
    target = torch.ones(1, 2, 2, dtype=torch.long)        # 全部是 rim
    loss, ce, dice = CEDiceLoss()(logits, target)
    # rim：(2·4/3 + ε)/(4/3 + 4 + ε) ≈ 0.5；cup：(0 + ε)/(4/3 + ε) ≈ 0 → dice loss ≈ 1 − 0.25
    assert abs(dice.item() - 0.75) < 1e-4
    assert abs(loss.item() - (0.5 * ce.item() + 0.5 * dice.item())) < 1e-6


def test_perfect_prediction():
    target = torch.tensor([[[0, 1], [2, 1]]])
    logits = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    loss, ce, dice = CEDiceLoss()(logits, target)
    assert ce.item() < 1e-6 and dice.item() < 1e-4


def test_no_foreground_is_finite():
    loss, _, _ = CEDiceLoss()(torch.randn(2, 3, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))
    assert torch.isfinite(loss)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('OK')
