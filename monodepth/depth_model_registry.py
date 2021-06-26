#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from monodepth.depth_model import DepthModel
from monodepth.midas_v2_model import MidasV2Model

from typing import List


def get_depth_model_list() -> List[str]:
    return ["midas2"]


def get_depth_model(type: str) -> DepthModel:
    if type == "midas2":
        return MidasV2Model
    else:
        raise ValueError(f"Unsupported model type '{type}'.")


def create_depth_model(type: str) -> DepthModel:
    model_class = get_depth_model(type)
    return model_class()
