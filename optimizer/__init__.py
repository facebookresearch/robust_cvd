#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from optimizer.radam import RAdam

OPTIMIZER_MAP = {
    "Adam": Adam,
    "RAdam": RAdam,
}


OPTIMIZER_NAMES = OPTIMIZER_MAP.keys()


OPTIMIZER_CLASSES = OPTIMIZER_MAP.values()


def create(optimizer_name: str, *args, **kwargs) -> Optimizer:
    return OPTIMIZER_MAP[optimizer_name](*args, **kwargs)
