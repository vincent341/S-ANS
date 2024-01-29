#visualize layout in /home/liushuang/program/OccupancyAnticipation_test/dat
# a/datasets/exploration/mp3d/v1/train/occant_gt_maps

import numpy as np
from PIL import Image
from glob import glob
import os
#seen_area_maps wall_maps
rootpath="../data/datasets/exploration/gibson/v1/train/occant_gt_maps/seen_area_maps"
outputpath = os.path.join(rootpath,"../my_layoutimg_seen/train")

if not os.path.exists(outputpath):
    os.makedirs(outputpath)
filelist=glob(os.path.join(rootpath,"*.npy"))

for filei in filelist:
    x=np.load(filei)
    name=os.path.split(filei)[-1][:-4]
    img=x[:,:,0]*255
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save(os.path.join(outputpath,name+".png"))
