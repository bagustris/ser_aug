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
├── iemocap-w2v2-svm-si
├── jtes-w2v2-svm-si
├── jtes-w2v2-svm-sti
├── jtes-w2v2-svm-ti
└── readme.md
```

# Installation
The scripts are written in python (tested on python version 3.8). The required packages are listed in requirements.txt. Please install them before running the scripts. We recommend using a virtual environment to install the packages.
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
# Caution
The code for reproducing ARHMM augmentations (and SPC, speech clenaed) are not 
given due copyrights. For SPC, you can use `librosa` library (using start-end silence 
removal, codes are not provided in this repository).

# Reproduce the results
1. To obtain results on JTES-SI (Table 2), please run the following command:
```
cd jtes-w2v2-svm-si
python jtest_si_preprocess_aug.py
python jtes_si_w2v2_svm_all.py
```

2. To obtain results on JTES-TI (Table 3), please run the following command:
```
cd jtes-w2v2-svm-ti
python jtes_ti_preprocess_aug.py
python jtes_ti_w2v2_svm_all.py
```

3. To obtain results on JTES-STI (Table 4), please run the following command:
```
cd jtes-w2v2-svm-sti
python jtes_sti_preprocess_aug.py
python jtes_sti_w2v2_svm_all.py
```

4. To obtain results on IEMOCAP-SI (Table 5), please run the following command:
```
cd iemocap-w2v2-svm  
python ieomocap_add_bg_ir_noi.py
python iemocap_preprocess_orig.py
python iemocap_preprocess_ir.py
python iemocap_preprocess_noi.py
python iemocap_preprocess_arhmm.py
python iemocap_w2v2_svm.py  
```

# Citation
Please cite the following paper if you use this repository:

> Atmaja, B. T.; Sasou, A. (2022). Effects of Data Augmentations 
> on Speech Emotion Recognition. Sensors, 22(16), 5941,
> https://doi.org/10.3390/s22165941.
