#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import cv2
from os.path import join as pjoin
import json
import math
import numpy as np
import torch.utils.data as data
import torch
from typing import List, Optional

from utils import image_io, frame_sampling as sampling

from lib_python import (DepthVideo, ValueXformType, DepthXformType, SpatialXformType)

_dtype = torch.float32


def load_image(
    path: str,
    channels_first: bool,
    check_channels: Optional[int] = None,
    post_proc_raw=lambda x: x,
    post_proc_other=lambda x: x,
) -> torch.FloatTensor:
    if os.path.splitext(path)[-1] == ".raw":
        im = image_io.load_raw_float32_image(path)
        im = post_proc_raw(im)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = post_proc_other(im)
    im = im.reshape(im.shape[:2] + (-1,))

    if check_channels is not None:
        assert (
            im.shape[-1] == check_channels
        ), "receive image of shape {} whose #channels != {}".format(
            im.shape, check_channels
        )

    if channels_first:
        im = im.transpose((2, 0, 1))
    # to torch
    return torch.tensor(im, dtype=_dtype)


def load_color(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        torch.tensor. color in range [0, 1]
    """
    im = load_image(
        path,
        channels_first,
        post_proc_raw=lambda im: im[..., [2, 1, 0]] if im.ndim == 3 else im,
        post_proc_other=lambda im: im / 255,
    )
    return im


def load_flow(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        flow tensor in pixels.
    """
    flow = load_image(path, channels_first, check_channels=2)
    return flow


def load_mask(path: str, channels_first: bool) -> torch.ByteTensor:
    """
    Returns:
        mask takes value 0 or 1
    """
    mask = load_image(path, channels_first, check_channels=1) > 0
    return mask.to(_dtype)


class VideoDataset(data.Dataset):
    """Load 3D video frames and related metadata for optimizing consistency loss.
    File organization of the corresponding 3D video dataset should be
        color_down/frame_{__ID__:06d}.raw
        flow/flow_{__REF_ID__:06d}_{__TGT_ID__:06d}.raw
        mask/mask_{__REF_ID__:06d}_{__TGT_ID__:06d}.png
        metadata.npz: {'extrinsics': (N, 3, 4), 'intrinsics': (N, 4)}
        <flow_list.json>: [[i, j], ...]
    """

    def __init__(
        self,
        path: str,
        frames: List[int],
        min_mask_ratio: float,
        use_temporal_smooth_loss: bool,
        meta_file: str = None,
        recon = None,
    ):
        """
        Args:
            path: folder path of the 3D video
        """
        self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.raw")
        if not os.path.isfile(self.color_fmt.format(0)):
            self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.png")

        self.mask_fmt = pjoin(path, "flow_mask", "mask_{:06d}_{:06d}.png")
        self.flow_fmt = pjoin(path, "flow", "flow_{:06d}_{:06d}.raw")

        self.recon = recon

        if meta_file is not None:
            with open(meta_file, "rb") as f:
                meta = np.load(f)
                self.extrinsics = torch.tensor(meta["extrinsics"], dtype=_dtype)
                self.intrinsics = torch.tensor(meta["intrinsics"], dtype=_dtype)
            assert (
                self.extrinsics.shape[0] == self.intrinsics.shape[0]
            ), "#extrinsics({}) != #intrinsics({})".format(
                self.extrinsics.shape[0], self.intrinsics.shape[0]
            )

        flow_list_fn = pjoin(path, "flow_list.json")
        if os.path.isfile(flow_list_fn):
            with open(flow_list_fn, "r") as f:
                self.flow_indices = json.load(f)

            # Strip header and remove pairs with low overlap.
            filtered = []
            for [frame0, frame1, score] in self.flow_indices[1:]:
                if (
                    frame0 in frames and frame1 in frames
                    and (min_mask_ratio is None or score > min_mask_ratio)
                ):
                    filtered.append([frame0, frame1])

            self.flow_indices = filtered
        else:
            names = os.listdir(os.path.dirname(self.flow_fmt))
            self.flow_indices = [
                self.parse_index_pair(name)
                for name in names
                if os.path.splitext(name)[-1] == os.path.splitext(self.flow_fmt)[-1]
            ]
            self.flow_indices = sampling.to_in_range(self.flow_indices)
        self.flow_indices = list(sampling.SamplePairs.to_one_way(self.flow_indices))

        self.use_temporal_smooth_loss = use_temporal_smooth_loss
        self.frames = frames
        self.num_frames = len(frames)

    def update_poses(self, depth_video: DepthVideo):
        """
        Update extrinsics, intrinsics, depth and spatial transformation
        from a depth_video. This should be called after every pose optimization.
        """

        N = depth_video.numFrames()
        ds_id = depth_video.numDepthStreams() - 1
        ds = depth_video.depthStream(ds_id)
        self.extrinsics = torch.zeros(N, 3, 4, dtype=_dtype)
        self.intrinsics = torch.zeros(N, 4, dtype=_dtype)
        self.scales = {}
        self.warp_map = {}

        valid_spatial_xform = \
            [SpatialXformType.Identity,
            SpatialXformType.VerticalLinear,
            SpatialXformType.CornersBilinear,
            SpatialXformType.BilinearGrid,
            SpatialXformType.BicubicGrid]

        for i in self.frames:
            f = ds.frame(i)

            # Update camera extrinsics
            self.extrinsics[i, :, 0] = torch.tensor(f.extrinsics.right())
            self.extrinsics[i, :, 1] = torch.tensor(f.extrinsics.up())
            self.extrinsics[i, :, 2] = torch.tensor(f.extrinsics.backward())
            self.extrinsics[i, :, 3] = torch.tensor(f.extrinsics.position)

            # Update camera intrinsics (expressed in pixels)
            W = ds.width() / 2.0
            H = ds.height() / 2.0
            self.intrinsics[i, 0] = W / math.tan(f.intrinsics.hFov / 2.0)
            self.intrinsics[i, 1] = H / math.tan(f.intrinsics.vFov / 2.0)
            self.intrinsics[i, 2] = W
            self.intrinsics[i, 3] = H

            # Update depth transformation
            xfDepth = f.depthXform()
            descDepth = xfDepth.desc()

            # We only support scale-based transforms at the moment.
            assert(descDepth.depthType == DepthXformType.Identity
                   or descDepth.valueXform == ValueXformType.Scale)

            # Either get a scalar scale or a map of scale parameters.
            if descDepth.depthType == DepthXformType.Identity:
                self.scales[i] = torch.Tensor([1.0]).reshape(1, 1)
            elif descDepth.depthType == DepthXformType.Global:
                self.scales[i] = torch.Tensor([xfDepth.params()[0]]).reshape(1, 1)
            elif descDepth.depthType == DepthXformType.Grid:
                self.scales[i] = torch.Tensor(xfDepth.paramMap(f)) # (H, W)
            else:
                raise RuntimeError(f"Unsupported depth transform type '{descDepth.type}'.")

            # Update spatial transformation
            xfSpatial = f.spatialXform()
            descSpatial = xfSpatial.desc()

            if descSpatial.spatialType in valid_spatial_xform:
                warp = xfSpatial.warp(ds.height(), ds.width())                  # (H, W, 2)
                self.warp_map[i] = torch.Tensor(np.transpose(warp, (2, 0, 1)))  # (2, H, W)
            else:
                raise RuntimeError(f"Unsupported spatial transform type '{descSpatial.type}'.")

    def parse_index_pair(self, name):
        strs = os.path.splitext(name)[0].split("_")[-2:]
        return [int(s) for s in strs]

    def get_neighbor_meta(self, frame_idx, img_shape):

        if 0 < frame_idx < self.num_frames - 1:
            anchor = [frame_idx, frame_idx]
            neighbors = [a + b for a, b in zip(anchor, [-1, 1])]
            flow_neighbor_pair = [[a, b] for a, b in zip(anchor, neighbors)]

            intrinsics_n = torch.stack([self.intrinsics[k] for k in neighbors], dim=0)
            extrinsics_n = torch.stack([self.extrinsics[k] for k in neighbors], dim=0)

            images_n = torch.stack(
                [load_color(self.color_fmt.format(k), channels_first=True)
                for k in neighbors],
                dim=0,
            )
            flows_n = [
                load_flow(self.flow_fmt.format(k_src, k_trg), channels_first=True)
                for k_src, k_trg in flow_neighbor_pair
            ]
            masks_n = [
                load_mask(self.mask_fmt.format(k_src, k_trg), channels_first=True)
                for k_src, k_trg in flow_neighbor_pair
            ]
        else:
            # Dummy variables (for making consistent tensor sizes across samples in a batch)
            N = 2
            H, W = img_shape
            intrinsics_n = torch.ones(N, 4)
            extrinsics_n = torch.ones(N, 3, 4)
            images_n = torch.zeros(N, 3, H, W)
            flows_n = [torch.ones(2, H, W) for i in range(N)]
            masks_n = [torch.ones(1, H, W) for i in range(N)]

        return (intrinsics_n, extrinsics_n, images_n, flows_n, masks_n)

    def __getitem__(self, index: int):
        """Fetch tuples of data. index = i * (i-1) / 2 + j, where i > j for pair (i,j)
        So [-1+sqrt(1+8k)]/2 < i <= [1+sqrt(1+8k))]/2, where k=index. So
            i = floor([1+sqrt(1+8k))]/2)
            j = k - i * (i - 1) / 2.

        The number of image frames fetched, N, is not the 1, but computed
        based on what kind of consistency to be measured.
        For instance, geometry_consistency_loss requires random pairs as samples -> N = 2
        When using temporal_smoothness_loss, it requires two sets of triplets -> N =6

        Returns:
            stacked_images (N, C, H, W): image frames
            targets: {
                'extrinsics': torch.tensor (N, 3, 4), # extrinsics of each frame.
                                Each (3, 4) = [R, t].
                                    point_wolrd = R * point_cam + t
                'intrinsics': torch.tensor (N, 4), # (fx, fy, cx, cy) for each frame
                'geometry_consistency':
                    {
                        'indices':  torch.tensor (2),
                                    indices for corresponding pairs
                                        [(ref_index, tgt_index), ...]
                        'flows':    ((2, H, W),) * 2 in pixels.
                                    For k in range(2) (ref or tgt),
                                        pixel p = pixels[indices[b, k]][:, i, j]
                                    correspond to
                                        p + flows[k][b, :, i, j]
                                    in frame indices[b, (k + 1) % 2].
                        'masks':    ((1, H, W),) * 2. Masks of valid flow matches
                                    to compute the consistency in training.
                                    Values are 0 or 1.
                    }
                'temporal_smoothness': (if using temporal smoothness scene flow loss)
                    {
                        'indices':  torch.tensor (4),
                                    indices for consecutive consecutive frames
                                        [(ref_index-1, ref_index + 1, tgt_index - 1, tgt_index + 1), ...]
                        'flows':    ((2, H, W),) * 4 in pixels.
                                    flows[0][b,:, i, j] - flow map for ref_index -> ref_index - 1 (backward flow)
                                    flows[1][b,:, i, j] - flow map for ref_index -> ref_index + 1 (forward flow)
                                    flows[2][b,:, i, j] - flow map for tgt_index -> tgt_index - 1 (backward flow)
                                    flows[3][b,:, i, j] - flow map for tgt_index -> tgt_index + 1 (forward flow)
                        'masks':    ((1, H, W),) * 4. Masks of valid flow matches
                                    to compute the consistency in training.
                                    Values are 0 or 1.
                        'valid':    torch.tensor (2), 1.0: valid; 0.0: invalid flow neighbors
                    }
            }

        """
        pair = self.flow_indices[index]

        indices = torch.tensor(pair)

        # Prepare metadata for the sampled frame pair
        intrinsics = torch.stack([self.intrinsics[k] for k in pair], dim=0)
        extrinsics = torch.stack([self.extrinsics[k] for k in pair], dim=0)

        images = torch.stack(
            [load_color(self.color_fmt.format(k), channels_first=True) for k in pair],
            dim=0,
        )
        flows = [
            load_flow(self.flow_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]
        masks = [
            load_mask(self.mask_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]

        metadata = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "geometry_consistency": {
                "indices": indices,
                "flows": flows,
                "masks": masks,
            },
        }

        if self.use_temporal_smooth_loss:
            ref_index, trg_index = pair
            _, _, H, W = images.shape

            # Get metadata from forward/backward frames
            intrinsics_n_ref, extrinsics_n_ref, images_n_ref, flows_n_ref, masks_n_ref = \
                self.get_neighbor_meta(ref_index, (H, W))
            intrinsics_n_trg, extrinsics_n_trg, images_n_trg, flows_n_trg, masks_n_trg = \
                self.get_neighbor_meta(trg_index, (H, W))

            # Concatentate the flow neighbors together
            intrinsics_n = torch.cat([intrinsics_n_ref, intrinsics_n_trg], dim=0)
            extrinsics_n = torch.cat([extrinsics_n_ref, extrinsics_n_trg], dim=0)
            images_n = torch.cat([images_n_ref, images_n_trg], dim=0)
            flows_n = flows_n_ref + flows_n_trg
            masks_n = masks_n_ref + masks_n_trg

            # Label invalid boundary frames
            valid_flow_neighbor = torch.zeros(2, 1)
            if 0 < ref_index < self.num_frames - 1:
                valid_flow_neighbor[0] = 1.0
            if 0 < trg_index < self.num_frames - 1:
                valid_flow_neighbor[1] = 1.0

            # Flow neigbhor frame indices
            anchor = [ref_index, ref_index, trg_index, trg_index]
            neighbors = [a + b for a, b in zip(anchor, [-1, 1, -1, 1])]
            neighbors = [max(0, min(idx, self.num_frames - 1)) for idx in neighbors]

            # Update the images, intrinsics, extrinsics, and metadata
            images = torch.cat([images, images_n], dim=0)
            intrinsics = torch.cat([intrinsics, intrinsics_n], dim=0)
            extrinsics = torch.cat([extrinsics, extrinsics_n], dim=0)

            # Update the extrinsics, intrinsics, and meta data for temporal smoothness
            metadata["extrinsics"] = extrinsics
            metadata["intrinsics"] = intrinsics
            metadata["temporal_smoothness"] = {
                "indices": torch.tensor(neighbors),
                "flows": flows_n,
                "masks": masks_n,
                "valid": valid_flow_neighbor,
            }

        N = 6 if self.use_temporal_smooth_loss else 2
        idx_list = list(pair) + neighbors if self.use_temporal_smooth_loss else pair

        # Prepare the depth scale (map)
        if getattr(self, "scales", None):
            if isinstance(self.scales, dict):
                metadata["scales"] = torch.stack([self.scales[k] for k in idx_list], dim=0)  # (N, 1, 1) or (N, H, W)
            else:
                metadata["scales"] = self.scales * torch.ones([N, 1], dtype=_dtype)

        # Prepare the 2D warp map from spatial transformation
        if self.recon != "colmap":
            metadata["warp"] = torch.stack([self.warp_map[k] for k in idx_list], dim=0)  # (N, 2, H, W)

        return (images, metadata)

    def __len__(self):
        return len(self.flow_indices)


class VideoFrameDataset(data.Dataset):
    """Load video frames from
        color_fmt.format(frame_id)
    """

    def __init__(self, color_fmt, frames=None):
        """
        Args:
            color_fmt: e.g., <video_dir>/frame_{:06d}.raw
        """
        self.color_fmt = color_fmt

        if frames is None:
            files = os.listdir(os.path.dirname(self.color_fmt))
            self.frames = range(len(files))
        else:
            self.frames = frames

    def __getitem__(self, index):
        """Fetch image frame.
        Returns:
            image (C, H, W): image frames
        """
        frame_id = self.frames[index]
        image = load_color(self.color_fmt.format(frame_id), channels_first=True)
        meta = {"frame_id": frame_id}
        return image, meta

    def __len__(self):
        return len(self.frames)
