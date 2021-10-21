from __future__ import print_function

import os
import numpy as np
import pandas as pd
from . import transforms
import torch
import torchvision
import unittest
import merlinpy
from .fastmri_dataloader_base import FastmriCartesianDatasetBase

class FastmriCartesianDataset(FastmriCartesianDatasetBase, torch.utils.data.Dataset):
    """MRI data set."""

    def __init__(
        self,
        config,
        mode,
    ):
        super().__init__(config, mode)
        self.transform = torchvision.transforms.Compose(transforms.get_torch_transform(mode, config))
        

    def __len__(self):
        return len(self.data_set)

    def _apply_transform(self, sample):
        return self.transform(sample)

class TestSingleCoilDataloader(unittest.TestCase):
    def _test(self, mode):
        import medutils

        path = os.path.dirname(os.path.realpath(__file__))
        config = merlinpy.loadYaml(f'{path}/config.yml', 'BaseExperiment')
        ds = FastmriCartesianDataset(config, mode=mode)
        if not 'test' in mode:
            inputs, outputs = ds.__getitem__(0)
            # extract inputs
            img_np = inputs[0][:,0].abs().numpy()
            kspace_np = inputs[1][:,0].abs().numpy()
            mask_np = inputs[2][:,0].numpy()

            ref = outputs[0][:,0]
            ref_np = ref.abs().numpy()
            medutils.visualization.imsave(medutils.visualization.plot_array(ref_np), f'{mode}_ref.png')
        else:
            inputs = ds.__getitem__(0)
            img_np = inputs['noisy'][:,0]
            kspace_np = inputs['kspace'][:,0]
            mask_np = inputs['mask'][:,0]
        
        kspace_np = np.fft.fftshift(kspace_np, (-2,-1))
        mask_np = np.fft.fftshift(mask_np, (-2,-1))

        medutils.visualization.imsave(medutils.visualization.plot_array(img_np),   f'{mode}_img.png')
        medutils.visualization.ksave(medutils.visualization.plot_array(kspace_np), f'{mode}_kspace.png')
        medutils.visualization.imsave(medutils.visualization.plot_array(mask_np),  f'{mode}_mask.png')

    def testSinglecoilTrain(self):
        self._test('singlecoil_train')

    def testSinglecoilVal(self):
        self._test('singlecoil_val')

    def testSinglecoilTest(self):
        self._test('singlecoil_test')

class TestMultiCoilDataloader(unittest.TestCase):
    def _test(self, mode):
        import medutils

        path = os.path.dirname(os.path.realpath(__file__))
        config = merlinpy.loadYaml(f'{path}/config.yml', 'BaseExperiment')
        ds = FastmriCartesianDataset(config, mode=mode)
        if not 'test' in mode:
            inputs, outputs = ds.__getitem__(0)
            # extract inputs
            img_np = inputs[0][:,0].abs().numpy()
            kspace_np = inputs[1].abs().numpy()
            mask_np = inputs[2].numpy()

            ref = outputs[0][:,0]
            ref_np = ref.numpy()
            ref_np = medutils.mri.rss(ref_np, 1)
            medutils.visualization.imsave(medutils.visualization.plot_array(ref_np),   f'{mode}_ref.png')
        else:
            inputs = ds.__getitem__(0)
            img_np = inputs['noisy'][:,0]
            kspace_np = inputs['kspace']
            mask_np = inputs['mask']
        
        kspace_np = np.fft.fftshift(kspace_np, (-2,-1))[:,0,0]
        mask_np = np.fft.fftshift(mask_np, (-2,-1))[:,0,0]

        img_np = medutils.mri.rss(img_np, 1)

        medutils.visualization.imsave(medutils.visualization.plot_array(img_np),   f'{mode}_img.png')
        medutils.visualization.ksave(medutils.visualization.plot_array(kspace_np), f'{mode}_kspace.png')
        medutils.visualization.imsave(medutils.visualization.plot_array(mask_np),  f'{mode}_mask.png')

    def testMulticoilTrain(self):
        self._test('multicoil_train')

    def testMulticoilVal(self):
        self._test('multicoil_val')

    def testMulticoilTest(self):
        self._test('multicoil_test')

if __name__ == '__main__':
    unittest.test()