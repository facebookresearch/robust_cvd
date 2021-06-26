#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn.functional as F


class ContrastLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @staticmethod
    def compute_ratio(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d_min = torch.min(x, y)
        d_max = torch.max(x, y)

        epsilon = 1e-10
        # return d_max / max(d_min, epsilon)
        return d_max / (d_min + epsilon)

    @classmethod
    def depth_ratio(cls, x: torch.Tensor):
        # gradient step=1
        right = F.pad(x, [0, 1, 0, 0])[:, :, 1:]
        bottom = F.pad(x, [0, 0, 0, 1])[:, 1:, :]

        ratio_h = cls.compute_ratio(right, x)  # x: left
        ratio_v = cls.compute_ratio(bottom, x)  # x: top

        # h_ratio will always have zeros in the last column, right-left
        # v_ratio will always have zeros in the last row,    bottom-top
        ratio_h[:, :, -1] = 0
        ratio_v[:, -1, :] = 0

        return (ratio_h, ratio_v)

    def forward(self, depth_orig: torch.Tensor, depth_pred: torch.Tensor):
        """Compute contrast loss.
        Args:
            depth_orig (B, N, H, W): Predicted depths from the original estimator.
            depth_pred (B, N, H, W): Currently predicted depth.
        """
        _, _, h, w = depth_pred.shape
        depth_orig = depth_orig.view(-1, h, w)
        depth_pred = depth_pred.view(-1, h, w)

        ratio_pred_h, ratio_pred_v = self.depth_ratio(depth_pred)
        ratio_orig_h, ratio_orig_v = self.depth_ratio(depth_orig)

        edge_map_h = ratio_orig_h > self.opt.lambda_contrast_thresh
        edge_map_v = ratio_orig_v > self.opt.lambda_contrast_thresh

        contrast_loss_h = torch.max(
            torch.square(self.opt.lambda_contrast_thresh - ratio_pred_h),
            torch.zeros_like(ratio_orig_h),
        )
        contrast_loss_h *= edge_map_h

        contrast_loss_v = torch.max(
            torch.square(self.opt.lambda_contrast_thresh - ratio_pred_v),
            torch.zeros_like(ratio_orig_v),
        )
        contrast_loss_v *= edge_map_v

        clv_num_zero = torch.sum(contrast_loss_v == 0) / (8 * 224 * 384)
        print(f"contrast_loss_v ratio of zero entries: {clv_num_zero}")
        print(f"contrast_loss_v max: {torch.max(contrast_loss_v)}")
        print(f"contrast_loss_v min: {torch.min(contrast_loss_v)}")

        # # Normalized contrast loss.
        # return torch.sum(contrast_loss_h) / torch.sum(contrast_loss_h != 0)
        # + torch.sum(contrast_loss_v) / torch.sum(contrast_loss_v != 0)
        # Summed contrast loss.
        contrast_loss = torch.sum(contrast_loss_h) / contrast_loss_h.shape[0] + \
        torch.sum(contrast_loss_v) / contrast_loss_v.shape[0]

        # Reweight the contrast loss.
        return self.opt.lambda_contrast_loss * contrast_loss
