import torch
import torch.nn.functional as F

def masked_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    assert reduction in ['none', 'mean']
    batchsize = input.size()[0]
    mask_norm = mask.sum((-2,-1), keepdim=True)
    mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))

    loss = F.l1_loss(input * mask, target * mask, reduction='none') / mask_norm

    if reduction == 'mean':
        loss = loss.view(batchsize, -1).sum(-1).mean()
    return loss

    
import unittest
import numpy as np

class TestLoss(unittest.TestCase):
    def _testLoss(self, mode):
        x = torch.from_numpy(np.random.randn(8,1,96,320))
        gt = torch.from_numpy(np.random.randn(*x.shape))
        mask = torch.from_numpy(np.ones_like(gt))
        loss = F.l1_loss(x, gt, reduction=mode)
        masked_loss = masked_l1_loss(x, gt, mask, reduction=mode)

    def testNone(self):
        self._testLoss('none')
    
    def testMean(self):
        self._testLoss('mean')