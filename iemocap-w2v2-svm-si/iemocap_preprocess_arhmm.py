# IEMOCAP extraction for au_ir
import os
from os.path import basename, splitext, join as path_join
import sys
import re
import json
# from librosa.util import find_files
from pathlib import Path
import glob

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'
AUG = 'spc'


def get_wav_paths(data_dirs):
    # for arhmm path is ended by either spc or glt
    wav_paths = glob.glob(data_dirs + '/**/*_spc.wav')
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        # start = wav_path.find('Session')
        # wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict


def preprocess(data_dirs, paths, out_path):
    meta_data = []
    for path in paths:
        print(f"path: {path}")
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(Path(data_dirs).parents[0], path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            print(f"label_path: {label_path}")
            with open(path_join(label_dir, label_path)) as f:
                for line in f:
                    # print(f"line: {line}")
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['neu', 'hap', 'ang', 'sad', 'exc']:
                        continue
                    # if line[1] not in wav_paths:
                    #     continue
                    # print(f"wav_paths[line[1]]: {line[1]}" + "_noi")
                    meta_data.append({
                        # change according to line 17 for arhmm
                        'path': wav_paths[line[1]+'_spc'],
                        'label': line[2].replace('exc', 'hap'),
                        # 'speaker': re.split('_', basename(wav_paths[line[1]]))[0]
                    })
    data = {
        'labels': {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3},
        'meta_data': meta_data
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)


def main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    # for i, path in enumerate(paths):
    os.makedirs(f"{out_dir}", exist_ok=True)
    preprocess(data_dir, paths[:4], path_join(f"{out_dir}", 'train_meta_data_spc.json'))

if __name__ == "__main__":
    main(sys.argv[1])
    # change line 50 based on the input argument
