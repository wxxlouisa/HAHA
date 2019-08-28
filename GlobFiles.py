import feat as featModule
import glob
import numpy as np
import os.path as osp
from collections import OrderedDict
import mmcv

fnList = glob.glob("/home/lab_speech/music_classification/fma_medium/*/*.mp3")
featDict = OrderedDict()
pbar = mmcv.ProgressBar(len(fnList))
for fn in fnList:
    sb = featModule.lowLevelExtractor(fn)
    sb0 = sb[0]['lowlevel.spectral_contrast_coeffs.mean']
    sb1 = sb[0]['lowlevel.spectral_contrast_coeffs.stdev']
    feat = np.concatenate([sb0, sb1])
    id_ = osp.basename(fn)
    featDict.update({id_:feat})
    pbar.update()
