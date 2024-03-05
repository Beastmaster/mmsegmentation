

import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
import sys
sys.path.insert(0, os.getcwd() + "/utils")
from data_utilities import *
from pathlib import Path
from PIL import Image
import cv2

class NiiSample:
    def __init__(self, img_name, mask_name):
        self.img_np = self._extract_nii(img_name, 3).astype(np.float32)
        self.mask_np = self._extract_nii(mask_name, 0).astype(np.uint8)
        self.shape = self.img_np.shape
        self.name = Path(img_name).stem.split('.')[0]

    def extract_sag(self):
        """
        inter_order = 0 for mask
        """
        for i in range(self.shape[2]):
            yield (i, self.img_np[:,:,i], self.mask_np[:, :, i])
    
    def extract_cor(self):
        for i in range(self.shape[2]):
            yield (i, self.img_np[:,i, :], self.mask_np[:, i, :])


    def _extract_nii(self, img_name, inter_order = 3):
        img_nib = nib.load(img_name)
        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=inter_order)
        img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
        im_np  = img_iso.get_fdata()
        print(im_np.dtype, np.max(im_np), np.min(im_np))
        return im_np


def SaveSample(sample, save_root, phase):
    """
    ### output format:
        # ├── data
        # │   ├── my_dataset
        # │   │   ├── img_dir
        # │   │   │   ├── train
        # │   │   │   │   ├── xxx{img_suffix}
        # │   │   │   │   ├── yyy{img_suffix}
        # │   │   │   │   ├── zzz{img_suffix}
        # │   │   │   ├── val
        # │   │   ├── ann_dir
        # │   │   │   ├── train
        # │   │   │   │   ├── xxx{seg_map_suffix}
        # │   │   │   │   ├── yyy{seg_map_suffix}
        # │   │   │   │   ├── zzz{seg_map_suffix}
        # │   │   │   ├── val

    """
    print(f"saving sample: {sample.name}")
    name = sample.name
    img_dir = Path(save_root) / "img_dir" / phase
    ann_dir = Path(save_root) / "ann_dir" / phase

    img_dir.mkdir(exist_ok = True, parents=True)
    ann_dir.mkdir(exist_ok = True, parents=True)
    
    # ret = sample.extract_sag()
    # print(ret)

    for i, sag_img, sag_mask in sample.extract_sag():
        img_name = img_dir / f"{sample.name}_{i}_sag.png"
        cv2.imwrite(str(img_name), sag_img)
        mask_name = ann_dir / f"{sample.name}_{i}_sag_mask.png"
        cv2.imwrite(str(mask_name), sag_mask)

# ver2019/sample/data/xx.nii.gz
# ver2019/sample/data/xx.nii.gz
source_root = "/Users/shuoqin/work/verse/data/"
data_name = "dataset-verse19validation"
save_root = "/Users/shuoqin/work/verse/data/extract"
phase = "train"
phase = "test"
# phase = "val"

raw_data_path = Path(source_root) / data_name / "rawdata"
mask_path = Path(source_root) / data_name / "derivatives"
# iterate dirs
for sub_dir in raw_data_path.iterdir():
    sub_name = sub_dir.name
    # print(sub_name)
    nii_files = list(sub_dir.glob("*.nii.gz"))
    if len(nii_files) < 1:
        continue
    img_nii = nii_files[0]
    print(img_nii.name.split('.')[0])
    mask_files = list((mask_path / sub_name).glob("*.nii.gz"))
    if len(mask_files) < 1:
        continue
    mask_nii = mask_files[0]
    data_sample = NiiSample(img_nii, mask_nii)
    SaveSample(data_sample, save_root, phase)
    break




