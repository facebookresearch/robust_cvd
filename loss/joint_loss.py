#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from typing import List, Optional

import torch
from torch.nn import Parameter

from loss.parameter_loss import ParameterLoss
from loss.consistency_loss import ConsistencyLoss
from loss.scene_flow_loss import SceneFlowLoss
from loss.disparity_smooth_loss import DisparitySmoothLoss
from loss.contrast_loss import ContrastLoss
from utils.torch_helpers import _device
from loaders.video_dataset import _dtype


class JointLoss(torch.nn.Module):
    def __init__(self, opt, parameters_init=None):
        super().__init__()
        self.opt = opt
        if opt.lambda_parameter > 0:
            assert parameters_init is not None
            self.parameter_loss = ParameterLoss(parameters_init, opt)

        if (
            opt.lambda_static_disparity > 0
            or opt.lambda_static_reprojection > 0
            or opt.lambda_static_depth_ratio > 0
        ):
            self.consistency_loss = ConsistencyLoss(opt)

        if (
            opt.lambda_scene_flow_static > 0
            or opt.lambda_smooth_reprojection > 0
            or opt.lambda_smooth_disparity > 0
            or opt.lambda_smooth_depth_ratio > 0
        ):
            self.scene_flow_loss = SceneFlowLoss(opt)

        if opt.lambda_disparity_smooth > 0:
            self.disparity_smooth_loss = DisparitySmoothLoss(opt)

        if opt.lambda_contrast_loss > 0:
            self.contrast_loss = ContrastLoss(opt)

    def __call__(
        self,
        images,
        depths_orig,
        depths,
        metadata,
        parameters: Optional[List[Parameter]] = None,
    ):
        loss = torch.zeros(1, dtype=_dtype, device=_device)
        batch_losses = {}
        if self.opt.lambda_parameter > 0:
            assert parameters is not None
            para_loss, para_batch_losses = self.parameter_loss(parameters)
            loss += para_loss
            print(f"===== para_loss: {para_loss} =====")
            batch_losses.update(para_batch_losses)

        if (
            self.opt.lambda_static_disparity > 0
            or self.opt.lambda_static_reprojection > 0
            or self.opt.lambda_static_depth_ratio > 0
        ):
            consis_loss, consis_batch_losses = self.consistency_loss(
                depths, metadata,
            )
            loss += consis_loss
            print(f"===== consis_loss: {consis_loss} =====")
            batch_losses.update(consis_batch_losses)

        scene_flow = None
        if (
            self.opt.lambda_scene_flow_static > 0
            or self.opt.lambda_smooth_disparity > 0
            or self.opt.lambda_smooth_reprojection > 0
            or self.opt.lambda_smooth_depth_ratio > 0
        ):
            scene_flow_loss, scene_flow_batch_losses, scene_flow = self.scene_flow_loss(
                depths, metadata,
            )
            loss += scene_flow_loss
            print(f"===== scene_flow_loss: {scene_flow_loss} =====")
            batch_losses.update(scene_flow_batch_losses)

        if self.opt.lambda_disparity_smooth > 0:
            disparity_smooth_loss, disparity_smooth_batch_losses = self.disparity_smooth_loss(
                images, depths,
            )
            loss += disparity_smooth_loss
            print(f"===== disparity_smooth_loss: {disparity_smooth_loss} =====")
            batch_losses.update(disparity_smooth_batch_losses)

        if self.opt.lambda_contrast_loss > 0:
            contrast_loss = self.contrast_loss(depths_orig, depths)
            loss += contrast_loss
            print(f"===== contrast_loss: {contrast_loss} =====")

        return loss, batch_losses, scene_flow
