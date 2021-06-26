#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from loss.distance import DIST_NAMES


class LossParams:
    """
    Loss related parameters
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--distance_type_static",
            default="l1",
            choices=DIST_NAMES,
            help="(robuset) distance function type",
        )
        parser.add_argument(
            "--distance_alpha",
            type=float,
            default=1,
            help="used when distance_type == general.",
        )
        parser.add_argument(
            "--distance_scale",
            type=float,
            default=1,
            help="used when distance_type is not l1.",
        )
        parser.add_argument(
            "--distance_type_smooth",
            default="l1",
            choices=DIST_NAMES,
            help="(robuset) distance function type",
        )
        parser.add_argument(
            "--lambda_static_disparity",
            type=float,
            default=0.0,
            help="weight for static disparity loss."
            " If < 0 it will be set automatically to the default for the"
            " specified model adapter.",
        )
        parser.add_argument(
            "--lambda_static_depth_ratio",
            type=float,
            default=100.0,
            help="weight for static depth ratio loss.",
        )
        parser.add_argument(
            "--lambda_static_reprojection",
            type=float,
            default=1.0,
            help="weight for static reprojection loss.",
        )
        parser.add_argument(
            "--lambda_scene_flow_static",
            type=float,
            default=0.0,
            help="weight for static scene flow loss.",
        )
        parser.add_argument(
            "--lambda_smooth_disparity",
            type=float,
            default=0.0,
            help="weight for temporally smooth scene flow loss (on disparity error).",
        )
        parser.add_argument(
            "--lambda_smooth_depth_ratio",
            type=float,
            default=0,
            help="weight for temporally smooth scene flow loss (on depth ratio error).",
        )
        parser.add_argument(
            "--lambda_smooth_reprojection",
            type=float,
            default=0.0,
            help="weight for temporally smooth scene flow loss (on reprojection error).",
        )
        parser.add_argument(
            "--lambda_parameter",
            type=float,
            default=0,
            help="weight for network parameter regularization loss.",
        )
        parser.add_argument(
            "--lambda_disparity_smooth",
            type=float,
            default=0.0,
            help="weight for spatial smoothness loss.",
        )
        parser.add_argument(
            "--sigma_color_grad",
            type=float,
            default=1.0,
            help="weight for spatial smoothness loss.",
        )
        parser.add_argument(
            "--lambda_contrast_thresh",
            type=float,
            default=1.05,
            help="depth ratio threshhold to compute contrast loss.",
        )
        parser.add_argument(
            "--lambda_contrast_loss",
            type=float,
            default=1.0,
            help="weight for the contrast loss.",
        )
        return parser

    @staticmethod
    def make_str(opt, exp_tag="short"):
        if exp_tag == "short":
            return (
                "StD{}".format(opt.lambda_static_depth_ratio)
                + "_StR{}".format(opt.lambda_static_reprojection)
                + "_SmD{}".format(opt.lambda_smooth_depth_ratio)
                + "_SmR{}".format(opt.lambda_smooth_reprojection)
            )
        else:
            return (
                "B{}".format(opt.lambda_static_disparity)
                + "_R{}".format(opt.lambda_static_reprojection)
                + "_St{}".format(opt.lambda_scene_flow_static)
                + "_Sm{}".format(opt.lambda_scene_flow_smooth)
                + "_Sp{}".format(opt.lambda_disparity_smooth)
                + (
                    "_{}".format(opt.distance_type)
                    + (
                        "-a{}".format(opt.distance_alpha)
                        if opt.distance_type == "general"
                        else ""
                    )
                    + (
                        "-c{}".format(opt.distance_scale)
                        if opt.distance_scale != 1
                        else ""
                    )
                )
                + "_PL1-{}".format(opt.lambda_parameter)
            )
