from __future__ import print_function

import os

import pandas as pd
import h5py

import medutils

import numpy as np
import torch

from torch.utils.data import Dataset

#import bart
from common import utils
from common.mytorch.tensor_ops import center_crop

import time


class LoadBatchedRecon(object):
    """ Temp class to load reconstructions and true acceleration factors"""
    def __init__(self, recon_path):
        self.recon_path = recon_path

    def __call__(self, sample):
        # load corresponding recon from dir
        recon_name = os.path.join(self.recon_path, sample['fname'].split('/')[-1])
        with h5py.File(recon_name, 'r') as recon_data:
            input = recon_data['reconstruction'][sample['slidx']]
            acc_true = recon_data['acceleration_true'][sample['slidx']]
            input_rss_mean = recon_data['input_rss_mean'][sample['slidx']]
            target_rss = recon_data['target_rss'][sample['slidx']]

        sample['input'] = input
        sample['acceleration_true'] = acc_true
        sample['acceleration'] = np.round(acc_true) # dummy
        sample['input_rss_mean'] = input_rss_mean
        sample['target_rss'] = target_rss
        return sample

class SENSE2RSSTrain(object):
    """ Temp class to perform SENSE2RSS. It loads recon, then normalise """
    def __call__(self, sample):
        norm = sample['attrs']['norm']
        def _batch_normalize(x):
            return x / norm.reshape(len(norm), *[1]*len(x.shape[1:]))

        input = _batch_normalize(sample['input'])
        target_rss = _batch_normalize(sample['target_rss'])
        sample['input_rss_mean'] = _batch_normalize(sample['input_rss_mean'])

        for key in ['norm', 'ref_max', 'mean', 'cov', 'rss_max']:
            if key in sample['attrs'].keys():
                sample['attrs'][key] = utils.numpy_to_torch(sample['attrs'][key]).to(torch.float32)

        sample_dict = {
            "input": utils.numpy_to_torch(input[:, np.newaxis].astype(np.float32)),
            "target_rss": utils.numpy_to_torch(target_rss[:, np.newaxis].astype(np.float32)),
            "attrs" : sample['attrs'],
            "fname" : sample['fname'],
            "slidx" : sample['slidx'],
            "acceleration" : utils.numpy_to_torch(sample['acceleration'].astype(np.float32)),
            "acceleration_true" : utils.numpy_to_torch(sample['acceleration_true'].astype(np.float32)),
            "input_rss_mean" : utils.numpy_to_torch(sample['input_rss_mean'].astype(np.float32)),
            "fg_mask" : center_crop(
                utils.numpy_to_torch(sample['fg_mask'].astype(np.float32)).unsqueeze_(1),
                (320, 320),
            )
        }

        return sample_dict

class SENSE2RSS(object):
    """ Temp class to perform SENSE2RSS. It loads recon, then normalise """
    def __init__(self, recon_path):
        self.recon_path = recon_path

    def __call__(self, sample):
        # load corresponding recon from dir
        recon_name = os.path.join(self.recon_path, sample['fname'].split('/')[-1])
        with h5py.File(recon_name, 'r') as recon_data:
            input = recon_data['reconstruction'][sample['slidx']]

        norm = sample['attrs']['norm']
        def _batch_normalize(x):
            return x / norm.reshape(len(norm), *[1]*len(x.shape[1:]))

        input = _batch_normalize(input)
        sample['input_rss_mean'] = _batch_normalize(sample['input_rss_mean'])

        for key in ['norm', 'ref_max', 'mean', 'cov', 'rss_max']:
            if key in sample['attrs'].keys():
                sample['attrs'][key] = utils.numpy_to_torch(sample['attrs'][key]).to(torch.float32)

        sample_dict = {
            "input": utils.numpy_to_torch(input[:, np.newaxis].astype(np.float32)),
            "attrs" : sample['attrs'],
            "fname" : sample['fname'],
            "slidx" : sample['slidx'],
            "acceleration" : utils.numpy_to_torch(sample['acceleration'].astype(np.float32)),
            "acceleration_true" : utils.numpy_to_torch(sample['acceleration_true'].astype(np.float32)),
            "input_rss_mean" : utils.numpy_to_torch(sample['input_rss_mean'].astype(np.float32)),
            "fg_mask" : center_crop(
                utils.numpy_to_torch(sample['fg_mask'].astype(np.float32)).unsqueeze_(1),
                (320, 320),
            )
        }

        return sample_dict


class MriBatchedPostProcessingDataset(Dataset):
    """MRI data set."""

    def __init__(self, csv_file, root_dir, batch_size, slices={}, data_filter={}, transform=None, norm='max', adjust_slice_samp_dist=False, full=False, acl=None, challenge='multicoil'):
        """
        Args:
            csv_file (string): Path to the csv data set descirption file.
            root_dir (string): Directory with all the data.
            data_filter (dict): Dict of filter options that should be applied to csv_file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.adjust_slice_samp_dist = adjust_slice_samp_dist
        self.full = full
        self.acl = acl
        self.challenge = challenge

        data_set = pd.read_csv(csv_file)

        # apply filter to data set
        for key in data_filter:
            if key != 'loc' and data_filter[key] != None:
                data_set = data_set[eval(f"(data_set.{key} == data_filter['{key}'])")]

        if 'loc' in data_filter:
            data_set = pd.DataFrame(data_set.loc[data_set.filename == data_filter['loc']])
        else:
            data_set = data_set[data_set.enc_y < 380]

        self.data_set = []
        self.full_data_set = []
        minsl = slices['min'] if 'min' in slices else 0
        for (fname, nPE) in zip(data_set.filename, data_set.nPE):
            h5_data = h5py.File(os.path.join(root_dir, fname), 'r')
            kspace = h5_data['kspace']
            num_slices = kspace.shape[0]
            maxsl = np.minimum(slices['max'], num_slices-1) if 'max' in slices else num_slices-1
            assert minsl <= maxsl
            #print(minsl, maxsl)
            assert isinstance(norm, str)
            attrs = {'nPE' : nPE, 'norm_str' : norm}
            #print(fname)
            self.data_set += [(fname, minsl, maxsl, attrs)] # skip the first ones
            self.full_data_set += [(fname, si, si, attrs) for si in range(minsl, maxsl)]
            h5_data.close()
        if self.full:
            self.data_set = self.full_data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]

        # give lower distribution on first 10 slices
        slice_range = np.arange(minsl, maxsl + 1)
        if self.adjust_slice_samp_dist:
            slice_prob = [1.] * 10 + [1.] * (maxsl + 1 - 10)
            slice_prob = np.array(slice_prob[minsl:])
            slice_prob /= slice_prob.sum()
        else:
            slice_prob = np.ones_like(slice_range, dtype=float)
            slice_prob /= slice_prob.sum()

        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))
        #choice = np.random.choice(np.arange(minsl, maxsl+1-self.batch_size))
        #slidx = list(np.arange(choice,choice+self.batch_size))
        #print(slidx)
        # load the kspace data for the given slidx
        # starttime = time.time()
        with h5py.File(os.path.join(self.root_dir, fname), 'r', libver='latest', swmr=True) as data:
            np_line = data['mask'].value if 'mask' in data.keys() else None
            np_acc = data.attrs['acceleration'] if 'acceleration' in data.attrs.keys() else None
            np_acl = data.attrs['num_low_frequency'] if 'num_low_frequency' in data.attrs.keys() else None

        sample = {
                  "line": np_line,
                  "acceleration": np_acc,
                  "acl": np_acl,
                  "attrs": attrs,
                  "slidx": slidx,
                  "fname": fname,
                  "rootdir": self.root_dir,
                 }

        if self.acl:
            sample['acl'] = self.acl

        if self.transform:
            sample = self.transform(sample)

        return sample

class MriBatchedPostProcessingDatasetEval(Dataset):
    """MRI data set."""

    def __init__(self, csv_file, root_dir, batch_size, slices={}, data_filter={}, transform=None, norm='max'):
        """
        Args:
            csv_file (string): Path to the csv data set descirption file.
            root_dir (string): Directory with all the data.
            data_filter (dict): Dict of filter options that should be applied to csv_file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size

        data_set = pd.read_csv(csv_file)

        # apply filter to data set
        for key in data_filter:
            if key != 'loc' and data_filter[key] not in [None]:
                data_set = data_set[eval(f"(data_set.{key} == data_filter['{key}'])")]
        if 'loc' in data_filter:
            data_set = pd.DataFrame(data_set.loc[data_set.filename == data_filter['loc']])

        data_set = data_set[data_set.enc_y < 380]

        self.data_set = []
        minsl = slices['min'] if 'min' in slices else 0
        for ii in range(len(data_set)):
            subj = data_set.iloc[ii]
            fname = subj.filename
            nPE = subj.nPE
            h5_data = h5py.File(os.path.join(root_dir, fname), 'r')
            kspace = h5_data['kspace']
            num_slices = kspace.shape[0]
            maxsl = np.minimum(slices['max'], num_slices-1) if 'max' in slices else num_slices-1
            assert minsl <= maxsl
            #print(minsl, maxsl)
            assert isinstance(norm, str)
            attrs = {'nPE' : nPE, 'norm_str' : norm, 'metadata': subj.to_dict() }
            #print(fname)
            self.data_set += [(fname, slidx, num_slices, attrs) for slidx in range(minsl, num_slices, self.batch_size)] # skip the first ones
            h5_data.close()

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        fname, slidx, num_slices, attrs = self.data_set[idx]

        slidx = list(np.arange(slidx,np.minimum(slidx+self.batch_size,num_slices)))
        with h5py.File(os.path.join(self.root_dir, fname), 'r', libver='latest', swmr=True) as data:
            np_acc = data.attrs['acceleration'] if 'acceleration' in data.attrs.keys() else None
            np_acl = data.attrs['num_low_frequency'] if 'num_low_frequency' in data.attrs.keys() else None

        # print("kspace loading took", time.time()-starttime)

        sample = {"acceleration": np_acc,
                  "acl": np_acl,
                  "attrs": attrs,
                  "slidx": slidx,
                  "fname": fname,
                  "rootdir": self.root_dir,
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample
