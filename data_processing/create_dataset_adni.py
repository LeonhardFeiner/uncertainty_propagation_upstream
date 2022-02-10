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
import xml.etree.ElementTree as ET
import json

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
     '--input-csv-file', type=pathlib.Path, required=True,
     help='Path to the csv file',
)

parser.add_argument(
     '--split-file', type=pathlib.Path, required=True,
     help='Path to the json file',
)

parser.add_argument(
     '--do-split', type=bool, default=False,
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


regex = re.compile(r"ADNI_(?P<first_part>\d{3}[_]S[_]\d{4})[_](?P<second_part>(?:\w|[-])*)[_](?P<third_part>\w*[_]\w*)\Z")


def complex_randn(*shape, random_gen = np.random):
    return random_gen.normal(size=(*shape, 2)).view(complex).squeeze(-1)

def to_snake_case(string):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()

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

def get_dataframe(xml_path):
    root = ET.parse(xml_path).getroot()
    project = root.find("project")
    subject = project.find("subject")
    study = subject.find("study")

    series = study.find("series")

    extract_subject = {
        "subjectIdentifier": str,
        "researchGroup": str,
    }
    extract_study = {
        "subjectAge": float,
        "ageQualifier": float,
        "ageQualifier": str,
        "weightKg": float,
        "postMortem": str,
    }
    extract_series = {
        "seriesIdentifier": str,
        "modality": str,
        "dateAcquired": pd.Timestamp,
    }

    def get_converted(node, key, converter):
        if node is None:
            return None
        else:
            found = node.find(key)
            if found is None:

                return None
            else:
                return converter(found.text)

    subject_dict = {
        key: get_converted(subject, key, converter)
        for key, converter in extract_subject.items()
    }

    study_dict = {
        key: get_converted(study, key, converter)
        for key, converter in extract_study.items()
    }

    series_dict = {
        key: get_converted(series, key, converter)
        for key, converter in extract_series.items()
    }

    
    image_path = get_file(xml_path)

    joint = {
        **subject_dict,
        **study_dict,
        **series_dict,
        "xml_path": xml_path,
        "image_path": image_path,
    }

    return {to_snake_case(key): value for key, value in joint.items()}


def get_diag(row):
    if (
        (row["DXCURREN"] == 1)
        or (row["DXCHANGE"] == 1)
        or (row["DXCHANGE"] == 7)
        or (row["DXCHANGE"] == 9)
        or (row["DIAGNOSIS"] == 1)
    ):
        return "CN"
    elif (
        (row["DXCURREN"] == 2)
        or (row["DXCHANGE"] == 2)
        or (row["DXCHANGE"] == 4)
        or (row["DXCHANGE"] == 8)
        or (row["DIAGNOSIS"] == 2)
    ):
        return "MCI"
    else:
        return "AD"

dataset_path = Path(args.data_path) / args.dataset 


input_df = pd.read_csv(args.input_csv_file)
xml_df = pd.DataFrame(
    [get_dataframe(xml_path) for xml_path in args.original_data_path.glob("*.xml")]
)

input_df["diagnosis_cleaned"] = input_df.apply(lambda row: get_diag(row), axis=1)
date_column_names = ["USERDATE", "USERDATE2", "EXAMDATE", "update_stamp"]
input_df[date_column_names] = input_df[date_column_names].applymap(pd.Timestamp)
df_merged = input_df.merge(
    xml_df, how="left", left_on="PTID", right_on="subject_identifier"
)

df_merged["time_delta"] = abs(df_merged.date_acquired - df_merged.EXAMDATE)

df_filtered = df_merged.iloc[df_merged.groupby("xml_path").time_delta.idxmin().tolist()]
df_filtered = df_filtered[df_filtered.time_delta.dt.days < 150]
if args.do_split:
    ptids = df_filtered.PTID.unique()
    rng = np.random.default_rng(0)
    rng.shuffle(ptids)

    split_positions = tuple((np.array([0.6, 0.8]) * len(ptids)).astype(int))
    train, val, test = np.split(ptids, split_positions)
    split_dict = dict(train=train.tolist(), val=val.tolist(), test=test.tolist())
    with open(Path(args.split_file).expanduser(), "w") as file:
        json.dump(split_dict, file)
else:
    with open(Path(args.split_file).expanduser(), "r") as file:
        split_dict = json.load(file)

for subset, id_list in split_dict.items():
    df_filtered.loc[df_filtered.PTID.isin(id_list),"subset"] = subset

assert 1 == df_filtered.groupby("PTID").subset.nunique().max()


for subset_name in split_dict.keys():
    df_subset = df_filtered[df_filtered.subset == subset_name]
# generate the file names
# image_names = sorted([os.path.join(args.dataset, f) for f in os.listdir(os.path.join(args.data_path, args.dataset)) if f.endswith(image_type)])

    # init dicts for infos that will be extracted from the dataset
    img_info = {'filename' : []}
    acq_info = {}
    seq_info = {}
    enc_info = {'nPE' : []}
    acc_info = {'acc' : [], 'num_low_freq' : []}

    subset_path = dataset_path / subset_name
    subset_path.mkdir(parents=True, exist_ok=True)


    for row in tqdm(df_subset.to_dict("records")):
        
        xml_path = row["xml_path"]
        image_path = row["image_path"]
        image = nib.load(image_path)

        array = np.squeeze(image.get_fdata(), -1).T
        noise = complex_randn(*array.shape, random_gen=rng) * (2 ** 6)
        transformed = fft2c(array) + noise
        
        with open(xml_path, "r") as f:
            xml = xmltodict.parse(f.read())


        enc_info['nPE'].append(transformed.shape[-1])
        acc_info['acc'].append(0)
        acc_info['num_low_freq'].append(0)

        file_name = subset_path / (xml_path.stem + ".h5")
        img_info["filename"].append(file_name)
        with h5py.File(file_name, "w") as hf:
            hf.create_dataset('kspace', data=transformed, compression="gzip")
            hf.create_dataset('reconstruction_esc', data=array, compression="gzip")

    data_info = {**img_info, **acq_info, **enc_info, **acc_info, **seq_info, **row}

    # convert to pandas
    df = pd.DataFrame(data_info)
    print(df)

    # save to output
    csv_file = args.csv_file.parent / f"{args.csv_file.name}_{subset_name}.csv"
    print(f'Save csv file to {csv_file}')
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
