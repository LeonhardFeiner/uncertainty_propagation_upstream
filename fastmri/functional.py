import torch
import torch.nn.functional as F
import medutils
from skimage.metrics import structural_similarity

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


def attenuated_masked_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask :torch.Tensor,
    log_b: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    assert reduction in ['none', 'mean']
    batchsize = input.size()[0]

    mask_norm = mask.sum((-2,-1), keepdim=True)
    mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))

    loss = (
        F.l1_loss(input * mask, target * mask, reduction='none') / mask_norm * torch.exp(-log_b)
        + log_b * 0.5 * mask / mask_norm
    )

    if reduction == 'mean':
        loss = loss.view(batchsize, -1).sum(-1).mean()
    return loss

def masked_l2_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    assert reduction in ['none', 'mean']
    batchsize = input.size()[0]
    mask_norm = mask.sum((-2,-1), keepdim=True)
    mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))

    loss = F.mse_loss(input * mask, target * mask, reduction='none') / mask_norm

    if reduction == 'mean':
        loss = loss.view(batchsize, -1).sum(-1).mean()
    return loss


def attenuated_masked_l2_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask :torch.Tensor,
    log_sigma_squared: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    assert reduction in ['none', 'mean']
    batchsize = input.size()[0]

    mask_norm = mask.sum((-2,-1), keepdim=True)
    mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))

    loss =  0.5 * (
        F.mse_loss(input * mask, target * mask, reduction='none') / mask_norm * torch.exp(-log_sigma_squared)
        + log_sigma_squared * mask / mask_norm
    )

    if reduction == 'mean':
        loss = loss.view(batchsize, -1).sum(-1).mean()
    return loss


def evaluate(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: tuple,
    mask = None,
):
    masked_input = (input * mask).detach().cpu().numpy()
    masked_target = (target * mask).detach().cpu().numpy()
    #dynamic_range = [target.min(), target.max()] if dynamic_range is None else dynamic_range

    # mse = F.mse_loss(masked_input, masked_target)
    # psnr = 10 * torch.log10(dynamic_range[1] ** 2 / mse)
    # psnr = torch.mean(psnr)
    mask_norm = mask.sum((-2,-1), keepdim=True)
    mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))


    nmse = medutils.measures.nmse(masked_input, masked_target, axes=dim) #* mask_norm
    l1 = (F.l1_loss(input * mask, target * mask, reduction='none') / mask_norm).view(input.shape[0], -1).sum(-1).mean()
    psnr = medutils.measures.psnr(masked_input, masked_target, axes=dim)# * mask_norm
    ssim_batch = [0 for i in range(input.shape[0])]
    for i in range(input.shape[0]):
        ssim_batch[i] = structural_similarity(masked_input[i,:].squeeze(), masked_target[i,:].squeeze(), axes=dim)
    ssim = np.mean(ssim_batch)

    return l1, nmse, psnr, ssim

    
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
    

class TestEval(unittest.TestCase):
    def _testEval(self, mode):
        x = torch.from_numpy(np.random.randn(8,1,96,320))
        gt = torch.from_numpy(np.random.randn(*x.shape))
        mask = torch.from_numpy(np.ones_like(gt))
        if mode == 'same':
            gt = x
        elif mode == 'minor':
            gt = x + 0.1
        elif mode == 'rand':
            gt = gt
        else:
            return NotImplemented
        return evaluate(x, gt, (-2,-1), mask)
    
    def testSame(self):
        self._testEval('same')
        self._testEval('rand')
        self._testEval('minor')