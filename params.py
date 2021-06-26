#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import torch

from monodepth.depth_model_registry import get_depth_model, get_depth_model_list
from depth_fine_tuning import DepthFineTuningParams
# from scale_calibration import ScaleCalibrationParams
# from tools.colmap_processor import COLMAPParams
# from tools.make_video import MakeVideoParams
from utils import frame_sampling, frame_range

from lib_python import DepthVideoPoseOptimizer
from utils.helpers import Nestedspace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Video3dParamsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--op",
            choices=["all", "extract_frames"], default="all")

        self.parser.add_argument("--path", type=str,
            help="Path to all the input (except for the video) and output files "
            " are stored.")

        self.parser.add_argument("--video_file", type=str,
            help="Path to input video file. Will be ignored if `color_full` and "
            "`frames.txt` are already present.")

        self.parser.add_argument("--recon",
            choices=["colmap", "i3d", "hd_depth"], default="i3d")

        self.parser.add_argument("--scaling",
            choices=["extrinsics", "depth"], default="depth")

        ## Temporally disable this flag due to an error poping up
        ## related to workflow parameter passing.
        # self.parser.add_argument("--configure",
        #     choices=["default", "kitti"], default="default")

        self.add_video_args()
        self.add_flow_args()
        # self.add_calibration_args()
        self.add_pose_optimization_args()
        self.add_fine_tuning_args()
        self.add_filter_args()
        self.add_saving_final_results_args()
        # self.add_export_args()

        self.initialized = True

    def add_video_args(self):
        self.parser.add_argument("--size", type=int, default=384,
            help="Size of the (long image dimension of the) output depth maps.")
        self.parser.add_argument("--short_side_target", action="store_true",
            help="Use short side of the image as the target size.")
        self.parser.add_argument("--align", type=int, default=32,
            help="Alignment requirement of the depth size (i.e, forcing each"
            " image dimension to be an integer multiple). If set <= 0 it will"
            " be set automatically, based on the requirements of the depth network.")

    def add_flow_args(self):
        self.parser.add_argument(
            "--flow_ops",
            nargs="*",
            help="optical flow operation: exhausted optical flow for all the pairs in"
            " dense_frame_range or consective that computes forward backward flow"
            " between consecutive frames.",
            choices=frame_sampling.SamplePairsMode.names(),
            default=["hierarchical2"],
        )
        self.parser.add_argument("--min_mask_ratio", type=float, default=0.2)
        self.parser.add_argument("--vis_flow", action="store_true")
        self.parser.add_argument("--flow_model", choices=["raft"], default="raft")

    # def add_calibration_args(self):
    #     COLMAPParams.add_arguments(self.parser)
    #     ScaleCalibrationParams.add_arguments(self.parser)

    def add_pose_optimization_args(self):
        defaults = DepthVideoPoseOptimizer.Params()
        self.parser.add_argument(
            "--opt.max_iterations", type=int, default=defaults.maxIterations)
        self.parser.add_argument(
            "--opt.num_threads", type=int, default=defaults.numThreads)
        self.parser.add_argument(
            "--opt.num_steps", type=int, default=defaults.numSteps)
        self.parser.add_argument(
            "--opt.robustness", type=float, default=defaults.robustness)
        self.parser.add_argument(
            "--opt.static_loss_type", type=str,
            choices=[
                "Euclidean",
                "ReproDisparity",
                "ReproDepthRatio",
                "ReproLogDepth"
            ],
            default="ReproDisparity")
        self.parser.add_argument(
            "--opt.static_spatial_weight", type=float, default=defaults.staticSpatialWeight)
        self.parser.add_argument(
            "--opt.static_depth_weight", type=float, default=defaults.staticDepthWeight)
        self.parser.add_argument(
            "--opt.smooth_loss_type",
            choices=[
                "EuclideanLaplacian",
                "ReproDisparityLaplacian",
                "ReproDepthRatioConsistency",
                "ReproLogDepthConsistency"
            ],
            default="ReproDisparityLaplacian")
        self.parser.add_argument(
            "--opt.smooth_static_weight", type=float, default=defaults.smoothStaticWeight)
        self.parser.add_argument(
            "--opt.smooth_dynamic_weight", type=float, default=defaults.smoothDynamicWeight)
        self.parser.add_argument(
            "--opt.position_regularization", type=float,
            default=defaults.positionReg)
        self.parser.add_argument(
            "--opt.scale_regularization", type=float,
            default=defaults.scaleReg)
        self.parser.add_argument(
            "--opt.scale_regularization_grid_size", type=int,
            default=defaults.scaleRegGridSize)
        self.parser.add_argument(
            "--opt.deformation_regularization_initial", type=float,
            default=defaults.depthDeformRegInitial)
        self.parser.add_argument(
            "--opt.deformation_regularization_final", type=float,
            default=defaults.depthDeformRegFinal)
        self.parser.add_argument(
            "--opt.adaptive_deformation_cost", type=float,
            default=defaults.adaptiveDeformationCost)
        self.parser.add_argument(
            "--opt.spatial_deformation_regularization", type=float,
            default=defaults.spatialDeformReg)
        self.parser.add_argument(
            "--opt.graduate_deformation_regularization", type=float,
            default=defaults.graduateDepthDeformReg)
        self.parser.add_argument(
            "--opt.focal_regularization", type=float,
            default=defaults.focalReg)
        self.parser.add_argument(
            "--opt.coarse_to_fine", type=str2bool, default=defaults.coarseToFine)
        self.parser.add_argument(
            "--opt.ctf_long", type=int, default=defaults.ctfLong)
        self.parser.add_argument(
            "--opt.ctf_short", type=int, default=defaults.ctfShort)
        self.parser.add_argument(
            "--opt.deferred_spatial_opt", type=str2bool, default=defaults.deferredSpatialOpt
        )
        self.parser.add_argument(
            "--opt.dso_long", type=int, default=defaults.dsoLong)
        self.parser.add_argument(
            "--opt.dso_short", type=int, default=defaults.dsoShort)
        self.parser.add_argument(
            "--opt.focal_long", type=float,
            default=defaults.focalLong)
        self.parser.add_argument(
            "--opt.intr_opt", type=str,
            choices=["Fixed", "Shared", "PerFrame"], default="PerFrame")
        self.parser.add_argument(
            "--opt.fix_poses", type=str2bool, default=defaults.fixPoses)
        self.parser.add_argument(
            "--opt.fix_depth_transforms", type=str2bool, default=defaults.fixDepthXforms)
        self.parser.add_argument(
            "--opt.fix_spatial_transforms", type=str2bool, default=defaults.fixSpatialXforms)
        self.parser.add_argument(
            "--opt.use_global_scale", action="store_true")
        self.parser.add_argument(
            "--opt.epipolar_dist_thresh", type=float, default=2.0)
        self.parser.add_argument(
            "--opt.dynamic_constraints", type=str,
            choices=["None", "Mask", "Ransac"], default="Mask")

    def add_fine_tuning_args(self):
        DepthFineTuningParams.add_arguments(self.parser)
        self.parser.add_argument(
            "--model_type", type=str, choices=get_depth_model_list(),
            default="midas2"
        )
        self.parser.add_argument(
            "--frame_range", default="",
            type=frame_range.parse_frame_range,
            help="Range of depth to fine-tune, e.g., 0,2-10,21-40."
        )
        self.parser.add_argument(
            "--exp_tag",
            type=str,
            choices=["short", "full"], default="short",
            help="Either short or long experiment names."
        )

    def add_filter_args(self):
        self.parser.add_argument("--post_filter", action="store_true")
        self.parser.add_argument("--filter_radius", type=int, default=4)

    def add_saving_final_results_args(self):
        self.parser.add_argument("--save_static", action="store_true")
        self.parser.add_argument("--save_finetuning", action="store_true")
        self.parser.add_argument("--save_vis", action="store_true")

    # def add_export_args(self):
    #     self.parser.add_argument("--render_depth_streams", type=str, nargs="+")
    #     self.parser.add_argument("--font_path", type=str)
    #     self.parser.add_argument("--renderer_shader_path", type=str)
    #     self.parser.add_argument("--viewer_shader_path", type=str)
    #     self.parser.add_argument("--effects_shader_path", type=str)
        # self.parser.add_argument("--make_video", action="store_true")
        # MakeVideoParams.add_arguments(self.parser)

    def parse(self, args=None, namespace=None):
        if not self.initialized:
            self.initialize()
        if not namespace:
            namespace = Nestedspace()

        # Change back from self.parser.parse_known_args(), to avoid silently
        # filtering the parameters with typos, and trigger exceptions instead.
        self.params = self.parser.parse_args(args, namespace=namespace)

        # if self.params.configure == "kitti":
        #     self.params.flow_checkpoint = "FlowNet2-KITTI"
        #     self.params.model_type = "monodepth2"
        #     self.params.overlap_ratio = 0.5
        #     if 'matcher' in self.params:
        #         self.params.matcher = 'sequential'

        # Resolve unspecified parameters
        model = get_depth_model(self.params.model_type)

        if self.params.align <= 0:
            self.params.align = model.align

        if self.params.learning_rate <= 0:
            self.params.learning_rate = model.learning_rate

        if self.params.lambda_static_disparity < 0:
            self.params.lambda_static_disparity = model.lambda_view_baseline

        # Multiply batch size by number of available GPUs.
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
        if num_gpus > 1:
            self.params.batch_size *= num_gpus
            print(f"Adjusting batch size to {self.params.batch_size}.")

        return self.params
