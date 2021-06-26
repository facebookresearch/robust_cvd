#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import cv2
import numpy as np
import os
import subprocess
import sys
import logging

from utils import image_io
from utils import colormaps


def visualize_scene_flow(scene_flow):
    """Visualize scene flow
        Args:
            scene_flow: a list of scene flow data
            - scene_flow[i]: a numpy array of size (B, 3, H, W)
            - 0: (ref -> trg); 1: (trg -> ref)
            - 2: (ref -> ref + 1); 3: (ref -> ref - 1)
            - 4: (trg -> trg + 1); 5: (trg -> trg - 1)
        Return:
            List of scene flow maps (in numpy)
    """

    scene_flow_vis = []

    for i in range(len(scene_flow)):
        # scene_flow_curr: (B, 3, H, W)
        scene_flow_curr = scene_flow[i]

        B, C, H, W = scene_flow_curr.shape
        scene_flow_curr = scene_flow_curr.reshape(B, -1)

        max_scene_flow_mag = np.max(np.absolute(scene_flow_curr), axis=1, keepdims=True)

        # Normalize scene flow to range [-1, 1]
        scene_flow_curr = scene_flow_curr / (max_scene_flow_mag + 1e-6)
        scene_flow_curr = scene_flow_curr.reshape(B, C, H, W)

        # Normalize scene flow to range [0, 1]
        scene_flow_curr = (scene_flow_curr + 1) / 2

        # Rescale the scene flow range [0, 255]
        scene_flow_curr_uint8 = np.uint8(scene_flow_curr * 255)

        scene_flow_vis.append(scene_flow_curr_uint8)

    return scene_flow_vis


def visualize_depth(depth, depth_min=None, depth_max=None):
    """Visualize the depth map with colormap.

    Rescales the values so that depth_min and depth_max map to 0 and 1,
    respectively.
    """
    if depth_min is None:
        depth_min = np.nanmin(depth)

    if depth_max is None:
        depth_max = np.nanmax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)

    return ((cv2.applyColorMap(
        depth_scaled_uint8, colormaps.cm_magma) / 255) ** 2.2) * 255


def visualize_depth_dir(
    src_dir: str, dst_dir: str, force: bool = False, extension: str = ".raw",
    min_percentile: float = 0, max_percentile: float = 100,
):
    src_files = []
    dst_files = []
    for file in sorted(os.listdir(src_dir)):
        base, ext = os.path.splitext(file)
        if ext.lower() == extension:
            src_files.append(file)
            dst_files.append(f"{base}.png")

    if len(src_files) == 0:
        return

    # Check if all dst_files already exist
    dst_exists = True
    for file in dst_files:
        if not os.path.exists(f"{dst_dir}/{file}"):
            dst_exists = False
            break

    if not force and dst_exists:
        return

    d_min = sys.float_info.max
    d_max = sys.float_info.min

    for src_file in src_files:
        print("reading '%s'." % src_file)
        if extension == ".raw":
            disparity = image_io.load_raw_float32_image(f"{src_dir}/{src_file}")
        else:
            disparity = cv2.imread(f"{src_dir}/{src_file}")
        ix = np.isfinite(disparity)

        if np.sum(ix) == 0:
            logging.warning(f"{src_file} has 0 valid depth")
            continue

        valid_disp = disparity[ix]
        d_min = min(d_min, np.percentile(valid_disp, min_percentile))
        d_max = max(d_max, np.percentile(valid_disp, max_percentile))

    for i in range(len(src_files)):
        src_file = src_files[i]
        dst_file = dst_files[i]

        print(f"reading '{src_file}'.")
        if os.path.exists(f"{dst_dir}/{dst_file}") and not force:
            print(f"skipping existing file '{dst_file}'.")
        else:
            if extension == ".raw":
                disparity = image_io.load_raw_float32_image(
                    f"{src_dir}/{src_file}")
            else:
                disparity = cv2.imread(f"{src_dir}/{src_file}")

            disparity_vis = visualize_depth(disparity, d_min, d_max)

            print(f"writing '{dst_file}'.")
            cv2.imwrite(f"{dst_dir}/{dst_file}", disparity_vis)


def create_video(pattern: str, output_file: str):
    ffmpeg = "/usr/local/fbprojects/fb-motion2/ffmpeg3-streaming2/bin/ffmpeg"
    if not os.path.exists(ffmpeg):
        print("ffmpeg not found. Install with 'sudo feature install fbmotion2'")
        sys.exit()
    if not os.path.exists(output_file):
        cmd = [ffmpeg, "-r", "30",
            "-i", pattern,
            "-c:v", "libx264",
            "-crf", "27",
            "-pix_fmt", "yuv420p",
            output_file]
        subprocess.call(cmd)


def apply_mask(im, mask, mask_color=None):
    im = im.reshape(im.shape[:2] + (-1,))
    C = im.shape[-1]
    mask = mask.reshape(mask.shape[:2] + (-1,)) > 0
    if mask_color is None:
        mask_color = np.array([0, 255, 0] if C == 3 else 1)
    mask_color = mask_color.reshape(1, 1, C)
    inv_mask = (1 - mask) * mask_color
    result = 0.7 * im + 0.3 * inv_mask
    return result.squeeze()
