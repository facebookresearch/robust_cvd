#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Suppress several verbose warnings for easier debugging.
import warnings  # isort:skip

# warnings.simplefilter("ignore", ResourceWarning)
# warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore")

import logging
import os
import os.path as osp
import sys
import multiprocessing as mp

sys.path.append(osp.abspath(__file__))
sys.path.append(osp.join(osp.dirname(__file__), "lib/build"))

print(sys.path)

import numpy as np
from iopath.common.file_io import g_pathmgr

from lib_python import (
    DepthVideo,
    FrameRange,
    initLib,
    logToStdout as glogToStdOut,
)

from depth_fine_tuning import DepthFineTuner
from flow import Flow
# from scale_calibration import calibrate_scale
from utils.frame_range import FrameRange, OptionalSet
from utils.helpers import (
    disable_output_stream_buffering,
    print_banner,
    print_namespace,
    print_subbanner,
    print_title,
)
from video import Video, sample_pairs
from dynamic_mask_generation import (
    DEFAULT_MASK_RCNN_CONFIG_PATH,
    dynamic_mask_generation,
    get_parser,
)

MODEL_DIR = "models/"

class DatasetProcessor:
    def __init__(self, params, writer=None, logToStdOut=False):
        # Init logging in C++ library.
        initLib()

        if logToStdOut:
            print("C++ logging to Stdout.")
            glogToStdOut()
        else:
            print("C++ *NOT* logging to Stdout.")

        # Prevent out-of-order logging when switching between python and C++.
        disable_output_stream_buffering()

        self.writer = writer
        self.params = params
        self.path = self.params.path

        print("------------ Parameters -------------")
        print_namespace(self.params)
        print("-------------------------------------")

        if self.params.video_file is not None:
            if not os.path.exists(self.params.video_file):
                sys.exit(f"ERROR: input video file {self.params.video_file} not found.")
            else:
                self.video_file = self.params.video_file
        else:
            self.video_file = None

    def create_output_path(self):
        range_tag = f"R{self.params.frame_range.name}"
        flow_ops_tag = "-".join(self.params.flow_ops)
        name = f"{range_tag}_{flow_ops_tag}_{self.params.model_type}"

        out_dir = osp.join(self.path, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def extract_frames(self):
        print_banner("Extracting original video")
        print_subbanner("PTS")
        self.video.extract_pts()

        print_subbanner("Video frames")
        self.video.extract_frames()

    def downscale_frames(self):
        print_banner("Downscaling")

        print_subbanner("Raw")
        self.video.downscale_frames(
            "color_down", self.params.size, "raw", align=self.params.align, short_side_target=self.params.short_side_target
        )

        print_subbanner("PNG")
        self.video.downscale_frames(
            "color_down_png", self.params.size, "png", align=self.params.align, short_side_target=self.params.short_side_target
        )

        print_subbanner("For flow")
        self.video.downscale_frames("color_flow", Flow.max_size(), "png", align=64)

    def compute_initial_depth(self):
        print_banner("Compute initial depth")
        initial_depth_dir = osp.join(self.path, f"depth_{self.params.model_type}")
        if not self.video.check_frames(osp.join(initial_depth_dir, "depth"), "raw"):
            ft = DepthFineTuner(
                self.out_dir, frames=None, base_dir=self.path, params=self.params
            )
            ft.save_depth(initial_depth_dir)

        return initial_depth_dir

    def compute_flow(self):
        print_banner("Compute flow")

        full_frame_range = FrameRange(
            frame_range=None, num_frames=self.video.frame_count
        )

        print_subbanner("Flow")
        frame_pairs = sample_pairs(full_frame_range, self.params.flow_ops)
        self.flow.compute_flow(frame_pairs, self.params.flow_model)

        print_subbanner("Masks")
        self.flow.compute_flow_masks()

        print_subbanner("Pair stats")
        self.flow.compute_flow_pair_stats(frame_pairs)

        if self.params.vis_flow:
            print_subbanner("Visualization")
            self.flow.visualize_flow(warp=True)

    def compute_dynamic_mask(self):
        print_banner("Compute Dynamic Mask")

        if self.video.check_frames(osp.join(self.path, "dynamic_mask"), "png"):
            print("Dynamic masks exist, checked OK.")
            return

        mp.set_start_method("spawn", force=True)
        try:
            color_dir = osp.join(self.path, "color_full")
            mask_dir = osp.join(self.path, "dynamic_mask")
            args, unknown = get_parser().parse_known_args()
            args.input = [color_dir + "/*.png"]
            args.output = mask_dir
            args.config_file = DEFAULT_MASK_RCNN_CONFIG_PATH
            # Generate masks with passed args
            dynamic_mask_generation(args)
        except Exception as e:
            logging.exception(e)

    def pipeline(self):
        self.extract_frames()
        self.downscale_frames()
        initial_depth_dir = self.compute_initial_depth()
        self.compute_flow()
        if self.params.opt.dynamic_constraints == "Mask":
           self.compute_dynamic_mask()

        frame_range = FrameRange(
            frame_range=self.params.frame_range.set, num_frames=self.video.frame_count
        )

        # if self.params.recon == "colmap":
        #     try:
        #         print_banner("Scale calibration")
        #         valid_frames = calibrate_scale(
        #             self.video, self.out_dir, frame_range, self.params
        #         )
        #         new_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        #         print(
        #             "Filtered out frames: ",
        #             sorted(set(frame_range.frames()) - set(new_frame_range.frames())),
        #         )

        #         print("Remaining valid frames: ", valid_frames)

        #         if len(new_frame_range.frames()) < len(frame_range.frames()):
        #             raise RuntimeError("COLMAP did not register all frames.")
        #     except Exception:
        #         print("Scale calibration failed. Skipping rest of pipeline.")
        #         return initial_depth_dir, None, frame_range.frames()

        print_banner("Fine-tuning")

        frames = frame_range.frames()

        print(f"frames: {frames}")

        # recon: colmap (CVD-1) or i3d (CVD-2)
        ft = DepthFineTuner(
            self.out_dir, frames=frames, base_dir=self.path, params=self.params
        )
        ft.fine_tune(writer=self.writer)

        # if self.params.render_depth_streams:
        #     print_banner("Rendering depth streams")
        #     self.render_depth_streams()

        # if self.params.make_video:
        #     print_banner("Export visualization videos")
        #     self.make_videos(ft.out_dir, frame_range)

        return initial_depth_dir, ft.out_dir, frame_range.frames()

    def process(self):
        os.makedirs(self.path, exist_ok=True)

        self.out_dir = self.create_output_path()

        self.video = Video(self.path, self.video_file)
        self.flow = Flow(self.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        if self.params.op == "all":
            return self.pipeline()
        elif self.params.op == "extract_frames":
            return self.extract_frames()
        else:
            raise RuntimeError("Invalid operation specified.")

        print("Done processing.")

    # def render_depth_streams(self):
    #     # Create OpenGL context.
    #     createGlContextForPython()

    #     # Load depth video.
    #     video = DepthVideo()
    #     video.load(self.path)
    #     video.printInfo()

    #     # Init renderer.
    #     renderer = Video3dRenderer(
    #         self.params.font_path,
    #         self.params.renderer_shader_path,
    #         self.params.viewer_shader_path,
    #         self.params.effects_shader_path,
    #     )

    #     renderParams = RenderParams()

    #     renderParams.depthRatioThresh = 1.05
    #     renderParams.acBack = 0.6
    #     renderParams.acUp = 0.0
    #     renderParams.acRight = 0.25
    #     renderParams.acLookAt = 0.7
    #     renderParams.cameraScale = 0.1

    #     if self.params.frame_range.set.set is not None:
    #         renderParams.framesSubset.fromString(str(self.params.frame_range.set)[1:-1])

    #     renderParams.displayWidth = video.width()
    #     renderParams.displayHeight = video.height()
    #     renderParams.displayAspect = video.aspect()

    #     renderParams.framesSubset.resolve(video.numFrames(), True)
    #     firstFrame = renderParams.framesSubset.firstFrame()

    #     exportParams = ExportParams()
    #     exportParams.pingPong = 0
    #     exportParams.outputVideo = 0

    #     # Export rendered videos.
    #     for ds_tag in self.params.render_depth_streams:
    #         try:
    #             ds_idx = -1
    #             if video.hasDepthStream(ds_tag):
    #                 ds_idx = video.depthStreamIndex(ds_tag)
    #             else:
    #                 ds_idx = int(ds_tag)
    #                 if ds_idx == -1:
    #                     ds_idx = video.numDepthStreams() - 1
    #                 if ds_idx < 0 or ds_idx >= video.numDepthStreams():
    #                     raise ValueError
    #         except ValueError:
    #             print(f"Could not render depth stream {ds_tag} --- not found.")
    #             continue

    #         print(f"Rendering depth stream {ds_idx}.")
    #         renderParams.depthStream = ds_idx

    #         ds = video.depthStream(ds_idx)
    #         print(f"Depth stream name: '{ds.name()}'.")

    #         df = ds.frame(firstFrame)
    #         renderParams.viewVFov = np.rad2deg(df.intrinsics.vFov)

    #         render_dir = osp.join(self.path, f"render/stream_{ds_idx:04d}")
    #         os.makedirs(render_dir, exist_ok=True)
    #         exportParams.outputPath = render_dir
    #         print(f"Render output directory: {exportParams.outputPath}")

    #         exportRenderedVideo(renderParams, exportParams, video, renderer)

    # def make_videos(self, ft_depth_dir, frame_range):
    #     args = [
    #         "--color_dir",
    #         osp.join(self.path, "color_down_png"),
    #         "--out_dir",
    #         osp.join(ft_depth_dir, "videos"),
    #         "--depth_labels",
    #         "MiDaS-v2",
    #         "CVD (initial)",
    #         "CVD (finetuned)",
    #         "--depth_dirs",
    #         osp.join(
    #             self.path, f"depth_{self.params.model_type}"
    #         ),  # Inital per-frame depth
    #     ]
    #     if self.params.recon == "colmap":
    #         args.append(osp.join(self.path, "depth_colmap_dense"))

    #     gt_dir = osp.join(self.path, "depth_gt")
    #     if os.path.isdir(gt_dir):
    #         args.append(gt_dir)

    #     vid_params = mkvid.MakeVideoParams().parser.parse_args(
    #         args, namespace=self.params
    #     )
    #     print("Make videos {}".format(vid_params))
    #     mkvid.main(vid_params)
