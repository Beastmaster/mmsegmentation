# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class VerSegDataset(BaseSegDataset):
    def __init__(self, img_suffix="_sag.png", seg_map_suffix="_sag_mask.png", **kwargs):
                super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=True,
            **kwargs)


