import json

filepath_orig = '/data/IEMOCAP_full_release/meta_data_aug/train_meta_data.json'
filepath_glt = '/data/IEMOCAP_full_release/arhmm/meta_data/train_meta_data_glt.json'
filepath_spc = '/data/IEMOCAP_full_release/arhmm/meta_data/train_meta_data_spc.json'
filepath_ir = '/data/IEMOCAP_full_release/aug_ir/meta_data/train_meta_data.json'
filepath_noi = '/data/IEMOCAP_full_release/aug_noise/meta_data/train_meta_data.json'


with open(filepath_orig, 'r') as f:
    data_train_orig = json.load(f)

with open(filepath_glt, 'r') as f:
    data_train_glt = json.load(f)

with open(filepath_spc, 'r') as f:
    data_train_spc = json.load(f)

with open(filepath_ir, 'r') as f:
    data_train_ir = json.load(f)

with open(filepath_noi, 'r') as f:
    data_train_noi = json.load(f)

# training paths
train_orig = [i['path'] for i in data_train_orig['meta_data']]
train_glt = [i['path'] for i in data_train_glt['meta_data']]
train_spc = [i['path'] for i in data_train_spc['meta_data']]
train_ir = [i['path'] for i in data_train_ir['meta_data']]
train_noi = [i['path'] for i in data_train_noi['meta_data']]

# labels
lab_orig = [i['label'] for i in data_train_orig['meta_data']]
lab_glt = [i['label'] for i in data_train_glt['meta_data']]
lab_spc = [i['label'] for i in data_train_spc['meta_data']]
lab_ir = [i['label'] for i in data_train_ir['meta_data']]
lab_noi = [i['label'] for i in data_train_noi['meta_data']]

# one augmentations
glt = train_orig + train_glt
spc = train_orig + train_spc
ir = train_orig + train_ir
noi = train_orig + train_noi

# labels for one augmentations
lab_orig_glt = lab_orig + lab_glt
lab_orig_spc = lab_orig + lab_spc
lab_orig_ir = lab_orig + lab_ir
lab_orig_noi = lab_orig + lab_noi

# two augmentations
glt_spc = glt + train_spc
spc_ir = spc + train_ir
ir_noi = ir + train_noi
noi_glt = noi + train_glt
glt_ir = glt + train_ir
spc_noi = spc + train_noi

# labels for two augmentation
lab_glt_spc = lab_orig_glt + lab_spc
lab_spc_ir = lab_orig_spc + lab_ir
lab_ir_noi = lab_orig_ir + lab_noi
lab_noi_glt = lab_orig_noi + lab_glt
lab_glt_ir = lab_orig_glt + lab_ir
lab_spc_noi = lab_orig_spc + lab_noi

# three augmentations
glt_spc_ir = glt_spc + train_ir
spc_ir_noi = spc_ir + train_noi
ir_noi_glt = ir_noi + train_glt
noi_glt_spc = noi_glt + train_spc

# labels for three augmentations
lab_glt_spc_ir = lab_glt_spc + lab_ir
lab_spc_ir_noi = lab_spc_ir + lab_noi
lab_ir_noi_glt = lab_ir_noi + lab_glt
lab_noi_glt_spc = lab_noi_glt + lab_spc

# four augmentations
glt_spc_ir_noi = glt_spc_ir + train_noi
lab_glt_spc_ir_noi = lab_glt_spc_ir + lab_noi

print(f"Length of glt_spc_ir_noi = {len(glt_spc_ir_noi)}")
print(f"Length of lab_glt_spc_ir_noi = {len(lab_glt_spc_ir_noi)}")

# save as csv
import pandas as pd
# pd.DataFrame(glt).to_csv('/data/IEMOCAP_full_release/meta_data_aug/' + 'glt.csv', index=False, header=None)

augs = [train_orig, glt, spc, ir, noi, glt_spc, spc_ir, ir_noi, noi_glt, glt_ir, spc_noi, glt_spc_ir, spc_ir_noi, ir_noi_glt, noi_glt_spc, glt_spc_ir_noi]

print(f"number of evaluated combinations: {len(augs)}")
for aug in augs:
    aug_name = [key for key, value in locals().items() if value == aug]
    print(aug_name[0])
    # print(aug)
    pd.DataFrame(aug).to_csv('/data/IEMOCAP_full_release/meta_data_aug/' 
        + str(aug_name[0]) + '.csv', index=False, header=None)

# for string is simpler
labels = ['lab_orig', 'lab_orig_glt', 'lab_orig_spc', 'lab_orig_ir', 'lab_orig_noi', 'lab_glt_spc', 'lab_spc_ir', 'lab_ir_noi', 'lab_noi_glt', 'lab_glt_ir', 'lab_spc_noi', 'lab_glt_spc_ir', 'lab_spc_ir_noi', 'lab_ir_noi_glt', 'lab_noi_glt_spc', 'lab_glt_spc_ir_noi']

for label in labels:
    # lab_name = [kunci for kunci, nilai in locals().items() if nilai == label]
    # print(i)
    print(label)
    pd.DataFrame(vars()[label]).to_csv('/data/IEMOCAP_full_release/meta_data_aug/' + str(label) + '.csv', index=False, header=None)