"""
Creation of dataset *.csv files for the fastmri dataset.

Copyright (c) 2019 Kerstin Hammernik <k.hammernik at imperial dot ac dot uk>
Department of Computing, Imperial College London, London, United Kingdom

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
from dataclasses import dataclass
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
     "--data-path", type=pathlib.Path, required=True,
     help="Path to the dataset",
)

parser.add_argument(
     "--original-data-path", type=pathlib.Path, required=True,
     help="Path to the dataset",
)


parser.add_argument(
     "--csv-file", type=pathlib.Path, required=True,
     help="Path to the csv file",
)

parser.add_argument(
     "--input-csv-file-dir", type=pathlib.Path, required=True,
     help="Path to the csv file",
)

parser.add_argument(
     "--split-file", type=pathlib.Path, required=True,
     help="Path to the json file",
)

parser.add_argument(
     "--do-split", type=bool, default=False,
     help="Path to the csv file",
)

parser.add_argument(
     "--dataset", type=str, required=True,
     help="Dataset for which the csv file should be generated."
)


# @dataclass
# class ReplaceArgs:
#     data_path = Path(os.environ["FASTMRI_ROOT"]).expanduser()
#     csv_file = Path("../datasets/singlecoil_adni").resolve()
#     dataset = "adni/data/singlecoil"
#     original_data_path = Path("~/datasets/adni/adni/ADNI").expanduser()
#     input_csv_file_dir = Path("~/datasets/adni").expanduser()
#     split_file = Path("../datasets/singlecoil_adni_split.json").resolve()
#     do_split = True

# args = ReplaceArgs()

args = parser.parse_args()


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

# %%
csv_dir = Path(args.input_csv_file_dir)

input_df = pd.read_csv(csv_dir / "DXSUM_PDXCONV_ADNIALL.csv")
volume_df = pd.read_csv(csv_dir / "FOXLABBSI_01_25_22.csv")
other_df = pd.read_csv(csv_dir / "idaSearch_2_18_2022.csv")

# %%
xml_df = pd.DataFrame(
    [get_dataframe(xml_path) for xml_path in args.original_data_path.glob("*.xml")]
)
other_df["Study Date"] = other_df["Study Date"].apply(pd.Timestamp)
date_column_names = ["USERDATE", "USERDATE2", "EXAMDATE", "update_stamp"]
input_df["diagnosis_cleaned"] = input_df.apply(lambda row: get_diag(row), axis=1)
input_df[date_column_names] = input_df[date_column_names].applymap(pd.Timestamp)

date_column_names_volume = ["EXAMDATE", "RUNDATE", "update_stamp"]
volume_df[date_column_names_volume] = volume_df[date_column_names_volume].applymap(pd.Timestamp)
# %%


df_merged = input_df.merge(
    xml_df, how="left", left_on="PTID", right_on="subject_identifier"
)

df_merged["time_delta"] = abs(df_merged.date_acquired - df_merged.EXAMDATE)

df_filtered = df_merged.iloc[df_merged.groupby("xml_path").time_delta.idxmin().tolist()]
df_filtered = df_filtered[df_filtered.time_delta.dt.days < 150]
# %%

df_merged2 = df_filtered.merge(other_df,left_on="PTID", right_on="Subject ID")
df_merged2["time_delta2"] = abs(df_merged2["Study Date"] - df_merged2.date_acquired)

df_filtered2 = df_merged2.iloc[df_merged2.groupby("xml_path").time_delta2.idxmin().tolist()]
df_filtered2 = df_filtered2[df_filtered2.time_delta2.dt.days < 150]

# %%
df_merged3 = df_filtered2.merge(volume_df, on=["RID", "VISCODE", "VISCODE2"],how="outer")
df_merged3["time_delta3"] = abs(df_merged3.EXAMDATE_x - df_merged3.EXAMDATE_y)
df_merged3["time_delta3b"] = df_merged3["time_delta3"].fillna(pd.Timedelta(days=365))
df_filtered3 = df_merged3.iloc[df_merged3.groupby("xml_path").time_delta3b.idxmin().tolist()]
# df_filtered3 = df_merged3.iloc[df_merged3.loc[:"EXAMDATE_y"].isna().sum(1).groupby(df_merged3["xml_path"]).idxmin().tolist()]
df_filtered3 = df_filtered3[(df_filtered3.time_delta3.dt.days < 150) | pd.isna(df_filtered3.time_delta3.dt.days)]


df_to_process = df_filtered3
# %%

ptids = df_to_process.PTID.unique()
rng = np.random.default_rng(0)
rng.shuffle(ptids)

splits = dict(train0=0.36,val0=0.04,train1=0.36,val1=0.04,test=0.2)
assert np.isclose(sum(splits.values()), 1)
split_positions = (np.cumsum(list(splits.values()))[:-1] * len(ptids)).astype(int)
split_id_arrays = np.split(ptids, split_positions)
split_id_lists = [id_array.tolist() for id_array in split_id_arrays]
split_dict = dict(zip(splits.keys(), split_id_lists))

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

for subset, id_list in split_dict.items():
    df_to_process.loc[df_to_process.PTID.isin(id_list),"subset"] = subset

assert 1 == df_to_process.groupby("PTID").subset.nunique().max()


# %%
for subset_name in split_dict.keys():
    df_subset = df_to_process[df_to_process.subset == subset_name]
    # init dicts for infos that will be extracted from the dataset
    img_info = {"filename" : [], "xml_path" : [], "image_path" : []}
    acq_info = {"has_nan":[], "has_zero_slices":[], "has_zero_only":[]}
    seq_info = {}
    enc_info = {"nPE" : []}
    acc_info = {"acc" : [], "num_low_freq" : []}

    nifti_info = {
        "spacing_x": [],
        "spacing_y": [],
        "spacing_z": [],
        "spacing_time": [], 
        "shape_x": [],
        "shape_y": [],
        "shape_z": [],
        "quatern_b": [],
        "quatern_c": [],
        "quatern_d": [],
        "qoffset_x": [],
        "qoffset_y": [],
        "qoffset_z": [],
    }

    subset_path = dataset_path / subset_name
    subset_path.mkdir(parents=True, exist_ok=True)


    for xml_path, image_path in tqdm(
        df_subset[["xml_path", "image_path"]].itertuples(index=False),
        desc=subset_name,
        total=len(df_subset),
    ):
        image = nib.load(image_path)
        spacing_x, spacing_y, spacing_z, spacing_time = image.header.get_zooms()
        shape_x, shape_y, shape_z, _ = image.shape
        quatern_b = image.header["quatern_b"].item()
        quatern_c = image.header["quatern_c"].item()
        quatern_d = image.header["quatern_d"].item()
        qoffset_x = image.header["qoffset_x"].item()
        qoffset_y = image.header["qoffset_y"].item()
        qoffset_z = image.header["qoffset_z"].item()

        nifti_info["spacing_x"].append(spacing_x)
        nifti_info["spacing_y"].append(spacing_y)
        nifti_info["spacing_z"].append(spacing_z)
        nifti_info["spacing_time"].append(spacing_time)
        nifti_info["shape_x"].append(shape_x)
        nifti_info["shape_y"].append(shape_y)
        nifti_info["shape_z"].append(shape_z)
        nifti_info["quatern_b"].append(quatern_b)
        nifti_info["quatern_c"].append(quatern_c)
        nifti_info["quatern_d"].append(quatern_d)
        nifti_info["qoffset_x"].append(qoffset_x)
        nifti_info["qoffset_y"].append(qoffset_y)
        nifti_info["qoffset_z"].append(qoffset_z)

        array = np.squeeze(image.get_fdata(), -1).T
        noise = complex_randn(*array.shape, random_gen=rng) * (2 ** 6)
        transformed = fft2c(array) + noise
        
        with open(xml_path, "r") as f:
            xml = xmltodict.parse(f.read())

        acq_info["has_nan"].append(np.isnan(array).any() or np.isnan(transformed).any())
        acq_info["has_zero_only"].append(np.max(array) <= 0) # or np.max(transformed) <= 0)
        acq_info["has_zero_slices"].append((np.max(array, (-2, -1)) <= 0).any()) # or np.max(transformed, (-2, -1)) <= 0)

        enc_info["nPE"].append(transformed.shape[-1])
        acc_info["acc"].append(0)
        acc_info["num_low_freq"].append(0)

        file_name = (xml_path.stem + ".h5")
        file_path = subset_path / (xml_path.stem + ".h5")
        img_info["xml_path"].append(xml_path.relative_to(args.data_path))
        img_info["image_path"].append(image_path.relative_to(args.data_path))
        img_info["filename"].append(file_path.relative_to(args.data_path))
        with h5py.File(file_path, "w") as hf:
            chunks = (1, *array.shape[1:])
            compression = "gzip"
            hf.create_dataset("kspace", data=transformed, chunks=chunks, compression=compression)
            hf.create_dataset(
                "reconstruction_esc", data=array, chunks=chunks, compression=compression
            )

    df_dict = df_subset.drop(columns=["xml_path", "image_path"]).to_dict("list")
    data_info = {**img_info, **nifti_info, **acq_info, **enc_info, **acc_info, **seq_info, **df_dict}

    # convert to pandas
    df = pd.DataFrame(data_info)
    print(df)

    # save to output
    csv_file = args.csv_file.parent / f"{args.csv_file.name}_{subset_name}.csv"
    print(f"Save csv file to {csv_file}")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)

# %%
