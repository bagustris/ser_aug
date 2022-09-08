# make IEMOCAP augmentation

import glob
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
# import numpy as np
import librosa
import os
import soundfile as sf
from pathlib import Path

noise_dir = '/home/bagus/research/2021/esc50_yamnet/datasets/ESC-50-master/audio/'
ir_dir = '/home/bagus/research/2021/jtes_make_aug/Audio/'

data_path = '/data/IEMOCAP_full_release/'
files = glob.glob(os.path.join(data_path + 'Session?/sentences/wav/*/', 
	'*.wav'))
files.sort()


augment_noise = Compose([AddBackgroundNoise(sounds_path=noise_dir)])
augment_ir = Compose([ApplyImpulseResponse(ir_path=ir_dir,
                     p=0.5,
                     lru_cache_size=128,
                     leave_length_unchanged=True)])


for i in files:
	# for debug, uncomment one line below, then comment the for loop
    #i = files[0]
    print(f"Processing ...{i}")
    data, sr = librosa.load(i, sr=16000, mono=True)
    data_noise = augment_noise(data, sample_rate=sr)
    data_ir = augment_ir(data, sample_rate=sr)
    extract_dir = Path(i).parts
    out_noi = os.path.join(data_path, 
                            'aug_noise', 
                            extract_dir[4], 
                            extract_dir[5])
    os.makedirs(out_noi, exist_ok=True)
    basename_noi = extract_dir[-1][:-4] + '_noi.wav'
    fn_noi = os.path.join(out_noi, basename_noi)
    out_imp = os.path.join(data_path, 
                            'aug_ir', 
                            extract_dir[4], 
                            extract_dir[5])
    os.makedirs(out_imp, exist_ok=True)
    basename_imp = extract_dir[-1][:-4] + '_imp.wav'
    fn_imp = os.path.join(out_imp, basename_imp)
    sf.write(fn_noi, data_noise, sr, subtype='PCM_24')
    sf.write(fn_imp, data_ir, sr, subtype='PCM_24')
        