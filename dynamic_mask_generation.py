#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import time

import cv2
import numpy as np
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from utils.predictor import VisualizationDemo


"""
- Pretrained weights (TODO(xrong): Replace with dropbox url):
    manifold://compphoto_data/tree/pretrained/
        first-party/mask-rcnn/mask_rcnn_R_50_FPN_3x.pkl
- Dynamic object categories in MSCOCO:
    Person + Vehicle + Animal:
    [person,
     bicycle, car, motorcycle, airplane, bus, train, truck, boat, bird, cat,
     dog, horse, sheep, cow, elephant, bear, zebra, giraffe]
    A more detailed chart can be checked here:
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
"""

DEFAULT_MASK_RCNN_MODEL_PATH = "models/mask_rcnn_R_50_FPN_3x.pkl"

DEFAULT_MASK_RCNN_CONFIG_PATH = "configs/mask_rcnn_R_50_FPN_3x.yaml"

# constants
WINDOW_NAME = "COCO detections"

# dynamic class id list
DYNAMIC_OBJECT_CATEGORIES = list(range(0, 8)) + list(range(13, 23))


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    # cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Dynamic mask generation CLI demo")
    parser.add_argument(
        "--config_file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video_input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dilation_factor",
        help="the factor to dilate the binary mask",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save_anno", action="store_true",
        help="save anonymized images",
    )
    return parser


def dynamic_mask_generation(args):
    local_model_path = DEFAULT_MASK_RCNN_MODEL_PATH
    cfg = setup_cfg(args)
    cfg.merge_from_list(["MODEL.WEIGHTS", local_model_path])
    cfg.freeze()

    demo = VisualizationDemo(cfg)

    if args.input:
        if args.input:
            print(f"dynamic frames input paths: {args.input}")
            args.input = glob.glob(osp.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            print(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if osp.isdir(args.output):
                    out_filename = osp.join(args.output, osp.basename(path))
                elif osp.isfile(args.output):
                    assert (
                        len(args.input) == 1
                    ), "Please specify a *directory* with args.output"
                    out_filename = args.output
                else:
                    os.makedirs(args.output, exist_ok=True)
                    out_filename = osp.join(args.output, osp.basename(path))

                # visualized_output.save(out_filename)
                mask_classes = predictions["instances"].get("pred_classes").cpu()
                mask_tensors = predictions["instances"].get("pred_masks").cpu()
                # the output masked image, similar to the anonymization output
                mask_img = np.transpose(np.copy(img), (2, 0, 1)).astype(np.uint8)
                # the output binary mask
                mask = np.zeros(img.shape[:2]).astype(np.uint8)

                # only mask out the dynamic object categories
                for idx, mask_class in enumerate(mask_classes):
                    if mask_class in DYNAMIC_OBJECT_CATEGORIES:
                        # get the category-specific mask
                        mask_tensor = mask_tensors[idx].numpy()
                        # aggregate category-specific mask to the output mask
                        mask[mask_tensor] = 255
                        # aggregate category-specific mask to the output masked image
                        for idx in range(3):
                            mask_img[idx][mask_tensor] = 255

                out_filename_prefix = osp.splitext(out_filename)[0]

                # save masked image
                if args.save_anno:
                    mask_img = np.transpose(mask_img, (1, 2, 0))
                    cv2.imwrite(out_filename_prefix + "_anon.png", mask_img)

                # save binary mask (invert to match the previous pipeline)
                mask = cv2.dilate(
                    mask,
                    kernel=np.ones(
                        (args.dilation_factor, args.dilation_factor), dtype=np.uint8
                    ),
                    iterations=1,
                )
                mask = cv2.bitwise_not(mask)
                cv2.imwrite(out_filename_prefix + ".png", mask)

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "Output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        assert args.input is None, "Cannot have both --input and --video_input!"
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = osp.basename(args.video_input)

        if args.output:
            if args.output.endswith((".mkv", ".mp4")):
                output_fname = args.output
            else:
                os.makedirs(args.output, exist_ok=True)
                output_fname = osp.join(args.output, basename)
                output_fname = osp.splitext(output_fname)[0] + ".mkv"
            assert not osp.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert osp.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print("Arguments: " + str(args))

    dynamic_mask_generation(args)
