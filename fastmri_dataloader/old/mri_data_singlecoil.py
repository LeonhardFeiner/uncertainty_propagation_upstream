from __future__ import print_function

import os
import h5py

import medutils
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from common import utils
from data.mri_data_multicoil import (
    _read_data,
    GenerateRandomFastMRIChallengeMask,
    LoadForegroundMask,
    MriDatasetEval,
    SetupRandomFastMRIChallengeMask,
    ToTensor,
)


class Unsqueeze():
    def __init__(self, dim, keys):
        self.dim = dim
        self.keys = keys

    def __call__(self, sample):
        for k in self.keys:
            sample[k] = sample[k].unsqueeze_(self.dim)
        return sample


class ComputeInit():
    def __call__(self, sample):
        np_kspace = sample["kspace"]
        np_mask = sample["mask"]

        # shift data
        np_mask = np.fft.ifftshift(np_mask, axes=(-2, -1))
        Ny, Nx = np_kspace.shape[-2:]
        x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        adjust = (-1) ** (x + y)
        np_kspace = np.fft.ifftshift(np_kspace, axes=(-2, -1)) * adjust

        # compute init
        np_kspace *= np_mask
        np_input = medutils.mri.ifft2(np_kspace)

        # extract norm
        norm = sample['attrs']['norm']

        def _batch_normalize(x):
            # match the shape of norm array
            return x / norm.reshape(len(norm), *[1] * len(x.shape[1:]))

        sample['mask'] = np_mask
        sample['input'] = _batch_normalize(np_input)
        sample['kspace'] = _batch_normalize(np_kspace)

        if 'input_rss_mean' in sample.keys():
            sample['input_rss_mean'] = _batch_normalize(
                sample['input_rss_mean'])

        sample['fg_mask_norm'] = np.sum(
            sample['fg_mask'],
            axis=tuple(range(1, len(sample['fg_mask'].shape))),
            keepdims=True,
        )
        normedsqrt = np.sqrt(
            np.maximum(1, sample['fg_mask_norm']) / (Ny * 368.))
        sample['fg_mask_normedsqrt'] = sample['fg_mask'] / normedsqrt

        if 'target' in sample.keys():
            sample['target'] = _batch_normalize(sample['target'])

        return sample


class ComputeBackgroundNormalization(object):
    def __call__(self, sample):
        np_kspace = sample["kspace_bg"]
        np_mask = sample["mask"]

        # compute rss input for bg correction
        np_input_rss = abs(medutils.mri.ifft2c(np_kspace * np_mask))
        # rescale wrt to *true* acceleration factor
        acc_true = sample['acceleration_true']
        np_input_rss *= np.sqrt(
            acc_true.reshape(
                len(acc_true), *[1]*len(np_input_rss.shape[1:])
            )
        )

        # extract 100x100 patch of the left upper corner of the first slice
        np_input_rss = medutils.mri.postprocess(np_input_rss)[..., :100, :]
        np_input_rss_mean = np.mean(
            np_input_rss, axis=tuple(np.arange(1, len(np_input_rss.shape))))
        sample['input_rss_mean'] = np_input_rss_mean
        return sample


class LoadStats(object):
    """Load Singlecoil stats
    """
    def __init__(self, attrs_dir):
        assert isinstance(attrs_dir, (str))
        self.attrs_dir = attrs_dir

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]

        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.attrs_dir, fname),
            'r',
            libver='latest',
            swmr=True,
        )
        attrs = h5_data.attrs
        np_target = []
        np_normval = []
        np_max = []
        np_mean = []
        np_cov = []
        norm = sample['attrs']['norm_str']

        rss_key = ['sos_max', 'rss_max'][0]  # temp fix
        is_testset = rss_key not in h5_data.attrs.keys()

        for i in range(sample['acceleration'].shape[0]):
            if 'acl' in sample.keys() and sample['acl'] is not None:
                acl = sample['acl']
            else:
                acl = 15 if sample['acceleration'][i] == 8 else 30

            # training and test data
            normval = attrs[f"{norm}_acl{acl}"]

            np_normval.append(normval)
            np_mean.append(attrs[f'lfimg_mean_acl{acl}'] / normval)
            np_cov.append(np.array(attrs[f'lfimg_cov_acl{acl}']) / normval)

            if not is_testset:
                np_target.append(utils.load_h5py_ensure_complex(
                    h5_data['reference'][sample["slidx"][i]]))
                np_max.append(attrs['sos_max'] / normval)

        h5_data.close()

        sample['attrs']['norm'] = np.array(np_normval)
        sample['attrs']['mean'] = np.array(np_mean)
        sample['attrs']['cov'] = np.array(np_cov)

        if not is_testset:
            sample['target'] = np.array(np_target)
            sample['attrs']['ref_max'] = np.array(np_max)

        return sample


class GeneratePatches(object):
    def __init__(self, patch_ny):
        self.patch_ny = patch_ny

    def __call__(self, sample):
        np_fg_mask = sample['fg_mask']
        np_kspace = sample['kspace']
        np_mask = sample['mask'][..., 0:self.patch_ny, :]

        # remove FE Oversampling
        np_fg_mask = medutils.mri.removeFEOversampling(
            np_fg_mask, axes=(-2, -1))

        # extract patch in Ny direction
        max_ny = np_fg_mask.shape[-2] - self.patch_ny + 1
        start_idx = np.random.randint(0, max_ny)
        start, end = start_idx, start_idx + self.patch_ny
        np_fg_mask = np_fg_mask[..., start:end, :]

        # coil-wise ifft of kspace, then patch extraction and coil-wise fft
        np_img = medutils.mri.ifft2c(np_kspace, axes=(-2, -1))
        np_img = medutils.mri.removeFEOversampling(np_img, axes=(-2, -1))
        np_img = np_img[..., start:end, :]
        np_kspace_new = medutils.mri.fft2c(np_img, axes=(-2, -1))
        np_target_new = np_img

        sample['target'] = np_target_new
        sample['kspace'] = np_kspace_new
        sample['mask'] = np_mask
        sample['fg_mask'] = np_fg_mask
        return sample


def create_eval_data_loaders(args, **kwargs):
    data_id = 'test_v2' if args.data_split == 'test' else args.data_split

    csv_val = kwargs.pop(
        'csv_eval', f'{args.csv_path}/multicoil_{data_id}.csv')
    data_filter = kwargs.pop('data_filter', {'type': args.acquisition})

    if args.data_split in ['train', 'val']:
        mask_func = GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=False,
        )
    else:
        mask_func = SetupRandomFastMRIChallengeMask()

    data_keys = ['input', 'target', 'kspace', 'mask']
    if args.mask_bg or args.use_stl:
        data_keys.append('input_rss_mean')

    data_transform = [
        mask_func,
        LoadStats(f'singlecoil_{data_id}_attrs'),
        LoadForegroundMask(f'singlecoil_{data_id}_foreground'),
        ComputeInit(),
        ToTensor(),
        Unsqueeze(1, data_keys),
    ]

    if args.mask_bg or args.use_stl:
        data_transform.insert(1, ComputeBackgroundNormalization())

    data = MriDatasetEval(
        csv_val,
        args.data_path,
        transform=transforms.Compose(data_transform),
        batch_size=args.batch_size,  # load all slices
        slices={},
        data_filter=data_filter,
        norm=args.norm,
        challenge='singlecoil',
    )

    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )

    return data_loader
