#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
from os.path import join as pjoin

import cv2
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr

import optical_flow_homography
from utils import consistency, flowlib, geometry, image_io, visualization
from utils.helpers import dotdict, mkdir_ifnotexists
from utils.torch_helpers import _device

RAFT_MODEL_PATH = "models/raft-things.pth"


def warp_by_flow(color, flow):
    def to_tensor(x):
        return torch.tensor(x.reshape((-1,) + x.shape)).to(_device).permute(0, 3, 1, 2)

    color = to_tensor(color)
    flow = to_tensor(flow)
    N, _, H, W = flow.shape
    pixel = geometry.pixel_grid(1, (H, W))
    uv = pixel + flow
    warped = geometry.sample(color, uv)
    return warped.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()


class Flow:
    def __init__(self, path, out_path):
        self.path = path
        self.out_path = out_path

    # Max size at which flow can be computed.
    @staticmethod
    def max_size():
        return 1024

    def compute_flow_pair_stats(self, frame_pairs):
        flow_list_path = pjoin(self.path, "flow_list.json")
        if os.path.isfile(flow_list_path):
            return flow_list_path

        def ratio(mask):
            return np.sum(mask > 0) / np.prod(mask.shape[:2])

        mask_fmt = pjoin(self.path, "flow_mask", "mask_{:06d}_{:06d}.png")
        results = [["frame0", "frame1", "mask_ratio"]]
        checked_pairs = set()
        for pair in frame_pairs:
            if pair in checked_pairs:
                continue
            cur_pairs = [pair, pair[::-1]]
            checked_pairs.update(cur_pairs)

            mask_fns = [mask_fmt.format(*ids) for ids in cur_pairs]
            masks = [cv2.imread(fn, 0) for fn in mask_fns]
            mask_ratios = [ratio(m) for m in masks]
            min_mask_ratio = min(mask_ratios)

            results.append([pair[0], pair[1], min_mask_ratio])
            results.append([pair[1], pair[0], min_mask_ratio])
            print(
                f"Frames {pair[0]} <-> {pair[1]}: "
                f"mask_ratio = {min_mask_ratio*100:04.1f}%"
            )

        with open(flow_list_path, "w") as f:
            json.dump(list(results), f)

    def check_flow_files(self, index_pairs):
        flow_dir = "%s/flow" % self.path
        for (i, j) in index_pairs:
            file = "%s/flow_%06d_%06d.raw" % (flow_dir, i, j)
            if not os.path.exists(file):
                return False
        return True

    def compute_flow(self, index_pairs, flow_model):
        if flow_model == "raft":
            model_file = RAFT_MODEL_PATH
        else:
            raise ValueError

        mkdir_ifnotexists("%s/flow" % self.path)

        if self.check_flow_files(index_pairs):
            return

        frame_dir = "%s/color_flow" % self.path
        frame1_fns = [
            "%s/frame_%06d.png" % (frame_dir, pair[0]) for pair in index_pairs
        ]
        frame2_fns = [
            "%s/frame_%06d.png" % (frame_dir, pair[1]) for pair in index_pairs
        ]
        out_fns = [
            "%s/flow/flow_%06d_%06d.raw" % (self.path, i, j) for (i, j) in index_pairs
        ]

        tmp = image_io.load_raw_float32_image(
            pjoin(self.path, "color_down", "frame_{:06d}.raw".format(0))
        )
        size = tmp.shape[:2][::-1]
        print("Resizing flow to", size)

        args = dotdict()
        args.model = flow_model
        args.pretrained_weights = model_file
        args.im1 = list(frame1_fns)
        args.im2 = list(frame2_fns)
        args.out = list(out_fns)
        args.size = size
        args.fp16 = False
        args.homography = True
        args.rgb_max = 255.0
        args.visualize = False

        optical_flow_homography.process(args)

        self.check_flow_files(index_pairs)

    def visualize_flow(self, warp=False):
        flow_fmt = pjoin(self.path, "flow", "flow_{:06d}_{:06d}.raw")
        mask_fmt = pjoin(self.path, "flow_mask", "mask_{:06d}_{:06d}.png")
        color_fmt = pjoin(self.path, "color_down", "frame_{:06d}.raw")
        vis_fmt = pjoin(self.path, "vis_flow", "frame_{:06d}_{:06d}.png")
        warp_fmt = pjoin(self.path, "vis_flow_warped", "frame_{:06d}_{:06d}_warped.png")

        def get_indices(name):
            strs = os.path.splitext(name)[0].split("_")[1:]
            return sorted((int(s) for s in strs))

        for fmt in (vis_fmt, warp_fmt):
            os.makedirs(os.path.dirname(fmt), exist_ok=True)

        flow_names = os.listdir(os.path.dirname(flow_fmt))
        for flow_name in flow_names:
            indices = get_indices(flow_name)
            if os.path.isfile(vis_fmt.format(*indices)) and (
                not warp or os.path.isfile(warp_fmt.format(*indices))
            ):
                continue

            indices_pair = [indices, indices[::-1]]
            flow_fns = [flow_fmt.format(*idxs) for idxs in indices_pair]
            mask_fns = [mask_fmt.format(*idxs) for idxs in indices_pair]
            color_fns = [color_fmt.format(idx) for idx in indices]

            flows = [image_io.load_raw_float32_image(fn) for fn in flow_fns]
            flow_ims = [flowlib.flow_to_image(np.copy(flow)) for flow in flows]
            colors = [image_io.load_raw_float32_image(fn) * 255 for fn in color_fns]
            masks = [cv2.imread(fn, 0) for fn in mask_fns]

            masked_colors = [
                visualization.apply_mask(im, mask) for im, mask in zip(colors, masks)
            ]
            masked_flows = [
                visualization.apply_mask(im, mask) for im, mask in zip(flow_ims, masks)
            ]

            masked = np.hstack(masked_colors + masked_flows)
            original = np.hstack(colors + flow_ims)
            visual = np.vstack((original, masked))
            cv2.imwrite(vis_fmt.format(*indices), visual)

            if warp:
                warped = [
                    warp_by_flow(color, flow)
                    for color, flow in zip(colors[::-1], flows)
                ]
                for idxs, im in zip([indices, indices[::-1]], warped):
                    cv2.imwrite(warp_fmt.format(*idxs), im)

    def compute_flow_masks(self, flow_thresh=1, color_thresh=1):
        flow_fmt = pjoin(self.path, "flow", "flow_{:06d}_{:06d}.raw")
        mask_fmt = pjoin(self.path, "flow_mask", "mask_{:06d}_{:06d}.png")
        color_fmt = pjoin(self.path, "color_down", "frame_{:06d}.raw")

        def get_indices(name):
            strs = os.path.splitext(name)[0].split("_")[1:]
            return [int(s) for s in strs]

        os.makedirs(os.path.dirname(mask_fmt), exist_ok=True)
        flow_names = os.listdir(os.path.dirname(flow_fmt))
        for flow_name in flow_names:
            indices = get_indices(flow_name)
            if os.path.isfile(mask_fmt.format(*indices)):
                continue

            indices_pair = [indices, indices[::-1]]
            flow_fns = [flow_fmt.format(*idxs) for idxs in indices_pair]
            mask_fns = [mask_fmt.format(*idxs) for idxs in indices_pair]
            color_fns = [color_fmt.format(idx) for idx in indices]

            flows = [image_io.load_raw_float32_image(fn) for fn in flow_fns]
            colors = [image_io.load_raw_float32_image(fn) for fn in color_fns]

            masks = consistency.consistent_flow_masks(
                flows, colors, flow_thresh, color_thresh
            )

            for mask, mask_fn in zip(masks, mask_fns):
                cv2.imwrite(mask_fn, mask * 255)
