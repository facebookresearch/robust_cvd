#!/usr/bin/env python3

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from iopath.common.file_io import PathManager
from PIL import Image

# from ..utils.image_io import save_raw_float32_image
from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images_L = []
    images_R = []

    for imfile in image_files:
        images_R.append(load_image(imfile))
        images_L.append(load_image(imfile.replace(".right.", ".")))

    images_L = torch.stack(images_L, dim=0)
    images_L = images_L.to(DEVICE)
    images_R = torch.stack(images_R, dim=0)
    images_R = images_R.to(DEVICE)

    padder = InputPadder(images_L.shape)

    return padder.pad(images_L)[0], padder.pad(images_R)[0]


def viz(img, flo, out):
    img = img[0].permute(1, 2, 0).cpu().numpy()

    # eliminate vertical flow values
    flo[0][1].fill_(0.0)
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    # save visualized flow map
    cv2.imwrite(out, flo[:, :, [2, 1, 0]])


def main(args):

    model = torch.nn.DataParallel(RAFT(args))

    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, "*.right.*")))
        print(f"== To process {len(images)} frame pairs ==")
        images_L, images_R = load_image_list(images)

        for i in range(images_L.shape[0]):
            img_L = images_L[i, None]
            img_R = images_R[i, None]

            # flow estimation
            flow_low, flow_up = model(img_L, img_R, iters=20, test_mode=True)
            # save visualized horizontal flow map
            flow_name = os.path.splitext(images[i])[0] + ".hflow.png"
            viz(img_L, flow_up, flow_name)

            # convert torch.Tensor to ndarray
            disp = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # use horizontal flow as disparities
            disp = disp[:, :, 0]
            disp_name = os.path.splitext(images[i])[0] + ".disp.npy"
            np.save(disp_name, disp)
            # save_raw_float32_image()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/raft-things.pth",
        help="restore checkpoint",
    )
    parser.add_argument("--path", help="dataset for evaluation")
    args = parser.parse_args()

    main(args)
