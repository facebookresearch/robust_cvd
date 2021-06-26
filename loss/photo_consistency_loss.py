#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn as nn
from typing import Dict, Tuple
from utils.geometry import sample, pixel_grid


def l1(x, dim):
    return torch.sum(torch.abs(x), dim=dim)


def mae(x, dim):
    return torch.mean(torch.abs(x), dim=dim)


def rmse(x, dim):
    return torch.norm(x, dim=dim)


class PhotoConsistencyLoss(nn.Module):
    def __init__(
        self,
        color_dist=lambda x, dim: torch.norm(x, dim=dim, keepdim=True),
        reduce: bool = True,
    ):
        """
        Args:
            color_dist: a function reduces C channel of color to a distance.
            Note color_dist should keep dim.
                It's called as
                    dist = self.color_dist(dist, dim=1)
            reduce: True if want to compute a scalar loss.
                False if want to compute spatial visualization of photo-consistency
        """
        super().__init__()
        self.color_dist = color_dist
        self.reduce = reduce

    def warp_field(self, depth: torch.Tensor, meta: Dict):
        raise NotImplementedError(
            "Warp is not implemented for PhotoConsistencyLoss"
            "Please call its child class"
        )

    def __call__(
        self, depth: torch.Tensor, image: torch.Tensor, image_t: torch.Tensor,
        meta: Dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            depth (B, 1, H, W): depth in reference view
            image (B, C, H, W): color/grayscale image in reference view
            image_t (B, C, H, W): color/grayscale image in target view
            meta: data necessary to create warp field that warps image_t to image
        Returns:
            if self.reduce:
                loss (1,), {'photo-consistency': batch_loss (B,)}
            else:
                loss (1, 1, H, W), {'photo-consistency': batch_loss (B, 1, H, W)}
        """
        assert image.shape == image_t.shape
        assert depth.shape[1] == 1
        assert depth.shape[0] == image.shape[0] and depth.shape[2:] == image.shape[2:]

        uv = self.warp_field(depth, meta)
        warped = sample(image_t, uv)

        ix = torch.isnan(depth).expand_as(warped)
        warped[ix] = float('nan')
        dist = self.color_dist(image - warped, dim=1)  # (B, C, H, W) -> (B, 1, H, W)

        if self.reduce:
            B = dist.shape[0]
            dist = torch.mean(dist.reshape(B, -1), dim=-1)  # (B, 1, H, W) -> B
        return torch.mean(dist, dim=0), {'photo-consistency': dist}


class RectifiedPhotoConsistencyLoss(PhotoConsistencyLoss):
    """
    Input image is rectified. So meta only contains baseline and horizontal focal_length
    """

    def warp_field(self, depth, meta: Dict):
        """
        depth (B, 1, H, W)
        meta['sign'] +- 1
        """
        sgn = meta['sign']
        disp = 1.0 / depth

        pix = pixel_grid(disp.shape[0], disp.shape[2:])
        u = pix[:, :1] + sgn * disp
        v = pix[:, 1:]
        uv = torch.cat((u, v), dim=1)
        return uv
