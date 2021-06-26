#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import os.path as osp
import sys

sys.path.append(osp.abspath(__file__))
sys.path.append(osp.join(osp.dirname(__file__), "lib/build"))
print(sys.path)

from params import Video3dParamsParser
from process import DatasetProcessor

if __name__ == "__main__":
    parser = Video3dParamsParser()
    params = parser.parse()

    dp = DatasetProcessor(params)
    dp.process()
