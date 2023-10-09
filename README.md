# Setup fastMRI environment
```
conda create --name fastmri
conda activate fastmri
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # haven't tested with cudatoolkit=11.3 yet...
# conda install pytorch=1.9.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia # that's what I use right now...
```

# Install requirements
```
pip install -r requirements.txt
```

# Environment variables
```
export FASTMRI_ROOT=<your-fastmri-root-dir>
```

# fastMRI setup
- `$FASTMRI_ROOT`
  - knee
    - data
  - adni
    - data

# Generate csv file
## FASTMRI
```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_train.csv --dataset knee/data/singlecoil_train  
```

```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_val.csv --dataset knee/data/singlecoil_val  
```

```
python ./data_processing/create_dataset_csv.py --data-path ${FASTMRI_ROOT} --csv-file=../datasets/singlecoil_knee_test.csv --dataset knee/data/singlecoil_test  
```

## ADNI
```
python ./data_processing/create_dataset_adni.py --data-path ${FASTMRI_ROOT} --dataset adni/data/singlecoil --original-data-path <ADNI Path> --input-csv-file <ADNI CSV Path> --split-file ./datasets/singlecoil_adni_split.json --csv-file ./datasets/singlecoil_adni 
```

ADNI Path must contain the image wise xml files and the image wise folders
ADNI CSV Path must contain "DXSUM_PDXCONV_ADNIALL.csv", "FOXLABBSI_*.csv", "idaSearch_*.csv" where * is replaced by a date

# Test dataloader
First, adapt the `<fastmri-root-dir>` in `fastmri_dataloader/config.yml`

```
python -m unittest fastmri_dataloader/fastmri_dataloader_th.py
```

# Train
rename `config_adni_train.yml` or `config_knee_train.yml` to `config.yml`

```
python train.py --regularizer Real2chCNN --num_workers 1 --aleatoric --l2 --save_checkpoint True --combined-extra-network --run-name multivariate_aleatoric
```

# Predict
rename `config_adni_predict.yml` or `config_knee_predict.yml` to `config.yml`
change both `accelerations:`  and `center_fractions` to a single value within `config.yml`


```
python test.py --regularizer Real2chCNN --aleatoric --l2 --non-diag-rank 8 --log-low-rank --ckpt_path <path to checkpoint>/checkpoint_epoch100.pth --cal-metrics --save-hdf5 --mode singlecoil_train1 --save-gnd --combined-extra-network --num_workers 8 
```

repeat for all accelerations