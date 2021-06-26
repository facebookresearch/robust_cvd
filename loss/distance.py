#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
from loss.general import lossfun
from utils.torch_helpers import _device
from loaders.video_dataset import _dtype


def create_general_dist(alpha, scale):
    alpha_t = torch.tensor(alpha, dtype=_dtype, device=_device)
    scale_t = torch.tensor(scale, dtype=_dtype, device=_device)
    return lambda x: lossfun(x, alpha_t, scale_t)


DIST_MAP = {
    "l1": lambda opt: lambda x: torch.abs(x / opt.distance_scale),
    "l2": lambda opt: create_general_dist(2, opt.distance_scale),
    "smooth_l1": lambda opt: create_general_dist(1, opt.distance_scale),
    "cauchy": lambda opt: create_general_dist(0, opt.distance_scale),
    "general": lambda opt: create_general_dist(opt.distance_alpha, opt.distance_scale),
}


DIST_NAMES = DIST_MAP.keys()


def create(distance_type, opt):
    """
    Create distance function of the following template:
        dist(x: torch.Tensor) -> torch.Tensor
    The returned tensor has the same shape as x.
    """
    return DIST_MAP[distance_type](opt)
