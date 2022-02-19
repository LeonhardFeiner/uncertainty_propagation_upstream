"""
Creation of dataset *.csv files for the fastmri dataset.

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
import os
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import xmltodict
import argparse
import pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
     '--data-path', type=pathlib.Path, required=True,
     help='Path to the dataset',
)
parser.add_argument(
     '--csv-file', type=pathlib.Path, required=True,
     help='Path to the csv file',
)
parser.add_argument(
     '--train-dataset', type=str, required=True,
     help='Dataset for which the csv file should be generated.'
)
parser.add_argument(
     "--train-annotations-file", type=pathlib.Path, required=True,
     help='Path to the csv file',
)

parser.add_argument(
     '--test-dataset', type=str, required=True,
     help='Dataset for which the csv file should be generated.'
)
parser.add_argument(
     "--test-annotations-file", type=pathlib.Path, required=True,
     help='Path to the csv file',
)

parser.add_argument(
     "--split-file", type=pathlib.Path, required=True,
     help='Path to the csv file',
)

parser.add_argument(
     "--do-split", type=bool, default=False,
     help="Path to the csv file",
)


args = parser.parse_args()


# @dataclass
# class ReplaceArgs:
#     data_path=Path(os.environ["FASTMRI_ROOT"])
#     csv_file = Path("../datasets/singlecoil_knee")
#     train_dataset = "knee/data/singlecoil_train" #knee/data/singlecoil_train 
#     train_annotations_file = Path("../datasets/kneeside_annotation_train.csv")
#     test_dataset = "knee/data/singlecoil_val" #knee/data/singlecoil_train 
#     test_annotations_file = Path("../datasets/kneeside_annotation_val.csv")
#     split_file=Path("../datasets/singlecoil_knee_split.json").resolve()
#     do_split=True

# args = ReplaceArgs()

image_type = '.h5'

# print(f'Create dataset file for {args.dataset}')
# output_name = f'{args.dataset}.csv'

label_df = pd.concat([
     pd.read_csv(args.train_annotations_file, index_col="file"),
     pd.read_csv(args.test_annotations_file, index_col="file"),
])
# generate the file names
all_train_image_names = sorted([
     os.path.join(args.train_dataset, f)
     for f in os.listdir(os.path.join(args.data_path, args.train_dataset))
     if f.endswith(image_type)
])

test_image_names = sorted([
     os.path.join(args.test_dataset, f)
     for f in os.listdir(os.path.join(args.data_path, args.test_dataset))
     if f.endswith(image_type)
])

# %%

rng = np.random.default_rng(0)
rng.shuffle(all_train_image_names)

splits = dict(train0=0.4, val0=0.1, train1=0.4, val1=0.1)
assert np.isclose(sum(splits.values()), 1)
split_positions = (np.cumsum(list(splits.values()))[:-1] * len(all_train_image_names)).astype(int)
split_id_arrays = np.split(all_train_image_names, split_positions)
split_id_lists = [id_array.tolist() for id_array in split_id_arrays]
split_dict = dict(zip(splits.keys(), split_id_lists))
split_dict["test"] = list(test_image_names)

if args.do_split:
    with open(Path(args.split_file).expanduser(), "w") as file:
        json.dump(split_dict, file)
else:
    original_split_dict = split_dict
    with open(Path(args.split_file).expanduser(), "r") as file:
        split_dict = json.load(file)

    equal_splits = ", ".join(
        key
        for key, id_list in split_dict.items()
        if np.array_equal(original_split_dict[key], id_list)
    )    
    print(f"splits {equal_splits or 'none'} would stay equal if splits were redone.")


for subset_name, image_names in split_dict.items():
     # init dicts for infos that will be extracted from the dataset
     img_info = {'filename' : image_names, 'acquisition' : []}
     acq_info = {'systemVendor' : [], 'systemModel' : [], 'systemFieldStrength_T' : [], 'receiverChannels' : [], 'institutionName' : [] }
     seq_info = {'TR' : [] , 'TE' : [], 'TI': [], 'flipAngle_deg': [], 'sequence_type': [], 'echo_spacing': []}
     enc_info = {'enc_x' : [], 'enc_y' : [], 'enc_z' : [], \
               'rec_x' : [], 'rec_y' : [], 'rec_z' : [], \
               'enc_x_mm' : [], 'enc_y_mm' : [], 'enc_z_mm' : [],
               'rec_x_mm' : [], 'rec_y_mm' : [], 'rec_z_mm' : [],
               'nPE' : []}
     acc_info = {'acc' : [], 'num_low_freq' : []}
     label_info = {"is_left_leg":[]}

     for fname in tqdm(image_names):
          full_path = args.data_path / fname
          label_info['is_left_leg'].append(label_df.loc[full_path.name])
          
          dset =  h5py.File(full_path,'r')
          acq = dset.attrs['acquisition']
          if acq == 'AXT1PRE': acq = 'AXT1'
          img_info['acquisition'].append(acq)
          acc_info['acc'].append(dset.attrs['acceleration'] if 'acceleration' in dset.attrs.keys() else 0)
          acc_info['num_low_freq'].append(dset.attrs['num_low_frequency'] if 'num_low_frequency' in dset.attrs.keys() else 0)
          header_xml = dset['ismrmrd_header'][()]
          header = xmltodict.parse(header_xml)['ismrmrdHeader']
          #pprint.pprint(header)   
          for key in acq_info.keys():
               acq_info[key].append(header['acquisitionSystemInformation'][key])
          for key in seq_info.keys():
               if key in header['sequenceParameters']:
                    seq_info[key].append(header['sequenceParameters'][key])
               else:
                    seq_info[key].append('n/a')
          enc_info['nPE'].append(int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['maximum'])+1)
          if int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['minimum']) != 0:
               raise ValueError('be careful!')
          for diridx in ['x', 'y', 'z']:
               enc_info[f'enc_{diridx}'].append(header['encoding']['encodedSpace']['matrixSize'][diridx])
               enc_info[f'rec_{diridx}'].append(header['encoding']['reconSpace']['matrixSize'][diridx])
               enc_info[f'enc_{diridx}_mm'].append(header['encoding']['encodedSpace']['fieldOfView_mm'][diridx])
               enc_info[f'rec_{diridx}_mm'].append(header['encoding']['reconSpace']['fieldOfView_mm'][diridx])

     data_info = {**img_info, **acq_info, **enc_info, **acc_info, **seq_info, **label_info}

     # convert to pandas
     df = pd.DataFrame(data_info)

     csv_file = args.csv_file.parent / f"{args.csv_file.name}_{subset_name}.csv"
     # save to output
     print(f'Save csv file to {csv_file}')
     csv_file.parent.mkdir(parents=True, exist_ok=True)
     df.to_csv(csv_file, index=False)
# %%