"""
Creation of dataset *.csv files for the fastmri dataset.

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from genericpath import exists
import os
import pandas as pd
import h5py
import xmltodict
import argparse
import pathlib
from pathlib import Path
import numpy as np
import nibabel as nib
import re
from tqdm import tqdm
from medutils.mri import fft2c

parser = argparse.ArgumentParser()
parser.add_argument(
     '--data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)

parser.add_argument(
     '--original-data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)


parser.add_argument(
     '--csv-file', type=pathlib.Path, required=True,
     help='Path to the csv file',
)
parser.add_argument(
     '--dataset', type=str, required=True,
     help='Dataset for which the csv file should be generated.'
)

args = parser.parse_args()

# image_type = '.h5'

# print(f'Create dataset file for {args.dataset}')
# output_name = f'{args.dataset}.csv'

def complex_randn(*shape, random_gen = np.random):
    return random_gen.normal(size=(*shape, 2)).view(complex).squeeze(-1)

regex = re.compile(r"ADNI_(?P<first_part>\d{3}[_]S[_]\d{4})[_](?P<second_part>(?:\w|[-])*)[_](?P<third_part>\w*[_]\w*)\Z")

def get_file(xml_path):
    m = regex.match(xml_path.stem)
    assert m is not None, (len(xml_path.stem), xml_path.stem)
    m.group("first_part"), m.group("second_part"), m.group("third_part")

    parent_path = Path(args.original_data_path) / m.group(1) / m.group(2)
    assert parent_path.exists(), parent_path
    file_name = f"ADNI_{m.group(1)}*{m.group(2)}*{m.group(3)}.nii"
    file_name = f"*{m.group(3)}.nii"
    nifti_path_list = list(parent_path.rglob(file_name))
    assert len(nifti_path_list) >= 1, (xml_path.stem, [f.stem for f in nifti_path_list])
    if len(nifti_path_list) > 1:
        print(len(nifti_path_list), parent_path.name)
    return nifti_path_list[0]

xmls = [path for path in args.original_data_path.glob("*.xml")]
niftis = [(get_file(xml_path), xml_path) for xml_path in xmls]
np.random.default_rng(0).shuffle(niftis)
split_positions = tuple((np.array([0.6, 0.8]) * len(niftis)).astype(int))

sets = dict(zip(("train", "val", "test"), np.split(niftis, split_positions)))

rng = np.random.default_rng(0)

for subset_name, nifti_tuples in sets.items():
# generate the file names
# image_names = sorted([os.path.join(args.dataset, f) for f in os.listdir(os.path.join(args.data_path, args.dataset)) if f.endswith(image_type)])

     # init dicts for infos that will be extracted from the dataset
     img_info = {'filename' : []}
     acq_info = {}
     seq_info = {}
     enc_info = {'nPE' : []}
     acc_info = {'acc' : [], 'num_low_freq' : []}

     subset_path = Path(args.data_path) / args.dataset / subset_name
     subset_path.mkdir(parents=True, exist_ok=True)


     for nifti_path, xml_path in tqdm(nifti_tuples):

          image = nib.load(nifti_path)

          array = np.squeeze(image.get_fdata(), -1).T
          noise = complex_randn(*array.shape, random_gen=rng) * (2 ** 4)
          transformed = fft2c(array) + noise
          
          with open(xml_path, "r") as f:
               xml = xmltodict.parse(f.read())


          acc_info['acc'].append(0)
          acc_info['num_low_freq'].append(0)

          file_name = subset_path / (xml_path.stem + ".h5")
          img_info["filename"].append(file_name)
          with h5py.File(file_name, "w") as hf:
               hf.create_dataset('kspace', data=transformed, compression="gzip")
               hf.create_dataset('reconstruction_esc', data=array, compression="gzip")

     data_info = {**img_info, **acq_info, **enc_info, **acc_info, **seq_info}

     # convert to pandas
     df = pd.DataFrame(data_info)
     print(df)

     # save to output
     csv_file = args.csv_file.parent / f"{args.csv_file.name}_{subset_name}.csv"
     print(f'Save csv file to {csv_file}')
     csv_file.parent.mkdir(parents=True, exist_ok=True)
     df.to_csv(csv_file, index=False)
