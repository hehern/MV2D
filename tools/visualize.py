import os
import tqdm
import json
from visual_nuscenes import NuScenes

use_gt = True
out_dir = './viz/nus/'
result_json = "..."
dataroot = 'data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.2)

# import ipdb; ipdb.set_trace()
# with open('{}.json'.format(result_json)) as f:
#     table = json.load(f)
# tokens = list(table['results'].keys())
tokens = nusc.sample

for token in tqdm.tqdm(tokens[::10]):#10个取一个
    if use_gt:
        nusc.render_sample(token['token'], out_path = out_dir+token['token']+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token['token'], out_path = out_dir+token['token']+"_pred.png", verbose=False)

