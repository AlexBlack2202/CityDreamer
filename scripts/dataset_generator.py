# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-31 15:04:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-28 10:55:26
# @Email:  root@haozhexie.com

import cv2
import logging
import numpy as np
import os
import sys
import torch

from PIL import Image

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

from extensions.extrude_tensor import TensorExtruder

# Global constants
HEIGHTS = {
    "ROAD": 4,
    "GREEN_LANDS": 8,
    "CONSTRUCTION": 10,
    "COAST_ZONES": 0,
    "ROOF": 1,
}
CLASSES = {
    "NULL": 0,
    "ROAD": 1,
    "BLD_FACADE": 2,
    "GREEN_LANDS": 3,
    "CONSTRUCTION": 4,
    "COAST_ZONES": 5,
    "OTHERS": 6,
    "BLD_ROOF": 7,
}
# NOTE: ID > 10 are reserved for building instances.
# Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
CONSTANTS = {
    "BLD_INS_LABEL_MIN": 10,
    "MAX_LAYOUT_HEIGHT": 640,
}


def get_instance_seg_map(seg_map, contours=None, use_contours=False):
    if use_contours:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (1 - contours).astype(np.uint8), connectivity=4
        )
    else:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (seg_map == CLASSES["BLD_FACADE"]).astype(np.uint8), connectivity=4
        )

    # Remove non-building instance masks
    labels[seg_map != CLASSES["BLD_FACADE"]] = 0
    # Building instance mask
    building_mask = labels != 0

    # Make building instance IDs are even numbers and start from 10
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
    labels = (labels + CONSTANTS["BLD_INS_LABEL_MIN"]) * 2

    seg_map[seg_map == CLASSES["BLD_FACADE"]] = 0
    seg_map = seg_map * (1 - building_mask) + labels * building_mask
    assert np.max(labels) < 2147483648
    return seg_map.astype(np.int32), stats[:, :4]


def get_seg_volume(part_seg_map, part_hf, tensor_extruder=None):
    if tensor_extruder is None:
        tensor_extruder = TensorExtruder(CONSTANTS["MAX_LAYOUT_HEIGHT"])

    seg_volume = tensor_extruder(
        torch.from_numpy(part_seg_map[None, None, ...]).cuda(),
        torch.from_numpy(part_hf[None, None, ...]).cuda(),
    ).squeeze()
    logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
    # Change the top-level voxel of the "Building Facade" to "Building Roof"
    roof_seg_map = part_seg_map.copy()
    non_roof_msk = part_seg_map <= CONSTANTS["BLD_INS_LABEL_MIN"]
    # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
    roof_seg_map = roof_seg_map - 1
    roof_seg_map[non_roof_msk] = 0
    for rh in range(1, HEIGHTS["ROOF"] + 1):
        seg_volume = seg_volume.scatter_(
            dim=2,
            index=torch.from_numpy(part_hf[..., None] + rh).long().cuda(),
            src=torch.from_numpy(roof_seg_map[..., None]).cuda(),
        )

    return seg_volume

