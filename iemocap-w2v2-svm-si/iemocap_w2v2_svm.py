import os

# import audb
# import audeer
# import audformat
import audinterface
import audonnx
import audmetric

# import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import json


# location of model, see: https://github.com/bagustris/w2v2-vad 
model_root = "/home/bagus/research/2022/sner_os_full/model"
# cache_root = "/home/bagus/research/2022/sner_os_full/cache"
model = audonnx.load(model_root)

# set augmentation here: 
# 0-aug: train_orig
# 1-aug: glt, spc, ir, noi, 
# 2-aug: glt_spc, spc_ir, ir_noi, noi_glt, glt_ir, spc_noi
# 3-aug: glt_spc_ir, spc_ir_noi, ir_noi_glt, glt_spc_noi, 
# 4-aug: glt_spc_ir_noi
# labels = ['lab_orig', 'lab_orig_glt', 'lab_orig_spc', 'lab_orig_ir', 'lab_orig_noi', 'lab_glt_spc', 'lab_spc_ir', 'lab_ir_noi', 'lab_noi_glt', 'lab_glt_ir', 'lab_spc_noi', 'lab_glt_spc_ir', 'lab_spc_ir_noi', 'lab_ir_noi_glt', 'lab_noi_glt_spc', 'lab_glt_spc_ir_noi']

# change the following two lines accordingly based on above
aug = 'train_orig'
lab_aug = 'lab_orig'

train_index = pd.read_csv(
    '/data/IEMOCAP_full_release/meta_data_aug/' + aug + '.csv',
    header=None) 
with open('/data/IEMOCAP_full_release/meta_data_aug/test_meta_data.json') as f:
    test_data = json.load(f)

train_index = train_index.stack().tolist()
train_emotion = pd.read_csv(
    '/data/IEMOCAP_full_release/meta_data_aug/' + lab_aug + '.csv',
    header=None)
train_emotion = train_emotion.stack().tolist()

test_index = [i['path'] for i in test_data['meta_data']]
test_emotion = [i['label'] for i in test_data['meta_data']]

# combine all data for feature extraction
all_index = train_index + test_index
all_emotion = train_emotion + test_emotion

# feature extraction
hidden_states = audinterface.Feature(
    model.outputs['hidden_states'].labels,
    process_func=model,
    process_func_args={
        'output_names': 'hidden_states',
    },
    sampling_rate=16000,    
    resample=True,    
    num_workers=5,
    # multiprocessing=True,
    verbose=True,
)

cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)


def cache_path(file):
    return os.path.join(cache_dir, file)


path = cache_path('w2v2_' + aug + '.pkl')
if not os.path.exists(path):
    features_w2v2 = hidden_states.process_files(
        all_index,
        # root=db.root,
    )
    features_w2v2.to_pickle(path)
    
features_w2v2 = pd.read_pickle(path)

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# create classifier and grouping object
clf = make_pipeline(
    StandardScaler(), 
    SVC(gamma='auto', verbose=True),
)


def experiment(features, targets):        
    # truths = []
    # preds = []
    
    train_x = features[:len(train_index)]
    train_y = targets[:len(train_index)]
    clf.fit(train_x, train_y)
    
    truth_x = features[len(train_index):]
    truth_y = targets[len(train_index):]
    predict_y = clf.predict(truth_x)
    
    # truths.append(truth_y)
    # preds.append(predict_y)
    truths = truth_y
    preds = predict_y

    # combine speaker folds
    truth = pd.Series(truths)
    truth.name = 'truth'
    pred = pd.Series(preds,
        index=truth.index,
        name='prediction',
    )
    
    return truth, pred


truth_w2v2, pred_w2v2 = experiment(
    features_w2v2,
    all_emotion) 

aur = audmetric.unweighted_average_recall(truth_w2v2, pred_w2v2)
wa = audmetric.accuracy(truth_w2v2, pred_w2v2)
print(f"Unweighted average recall {aug}: {aur}")
print(f"Weighted average recall orig {aug}: {wa} ")
