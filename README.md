# ser_aug
Repository for paper "Effect of data augmentation on SER"

~~Waiting for NEDO approval, once obtained, the codes will be hosted here.~~

# Introduction
This repository consists of python scripts to run experiments in the paper "Effects of Data Augmentations for Speech Emotion Recognitions".

Note that the datasets (IEMOCAP and JTES) are not included in this repository. Please obtain them from the authors of the dataset.

There are four experiments: JTES-SI, JTES-TI, JTES-STI, and IEMOCAP. Please run each script in each directory. The corresponding directories are 

```bash
bagus@m049:ser_aug$ tree
.
├── iemocap-w2v2-svm
├── jtes-w2v2-svm-si
├── jtes-w2v2-svm-sti
├── jtes-w2v2-svm-ti
└── readme.md
```

# Installation
The scripts are written in python (tested on python version 3.8). The required packages are listed in requirements.txt. Please install them before running the scripts. We recommend using virtual environment to install the packages.
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reproduce the results
1. To obtain results on JTES-SI (Table 2), please run the following command:
cd jtes-w2v2-svm-si
python jtest_si_preprocess_aug.py
python jtes_si_w2v2_svm_all.py

2. To obtain results on JTES-TI (Table 3), please run the following command:
cd jtes-w2v2-svm-ti
python jtes_ti_preprocess_aug.py
python jtes_ti_w2v2_svm_all.py

3. To obtain results on JTES-STI (Table 4), please run the following command:
cd jtes-w2v2-svm-sti
python jtes_sti_preprocess_aug.py
python jtes_sti_w2v2_svm_all.py

4. To obtain results on IEMOCAP-SI (Table 5), please run the following command:
cd iemocap-w2v2-svm
python iemocap_preprocess_aug.py
python iemocap_w2v2_svm_all.py

# Citation
Please cite the following paper if you use this repository:

> Atmaja, B. T.; Sasou, A. (2022). Effects of Data Augmentations 
> on Speech Emotion Recognition. Sensors, 22(16), 5941,
> https://doi.org/10.3390/s22165941.
