#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from cv2 import CV_32FC3, CV_8UC1
import os
from typing import List
import shutil

from lib_python import (
    DepthVideo,
    DepthVideoImporter,
    DepthVideoPoseOptimizer,
    DepthVideoProcessor,
    DepthXformType,
    FlowConstraintsCollection,
    FlowConstraintsParams,
    IntrinsicsOptimization,
    SmoothLossType,
    SpatialXformType,
    StaticLossType,
    ValueXformType,
    XformType,
)

from utils.helpers import Nestedspace


def convert_opt_params(src: Nestedspace) -> DepthVideoPoseOptimizer.Params:
    dst = DepthVideoPoseOptimizer.Params()
    dst.maxIterations = src.max_iterations
    dst.numThreads = src.num_threads
    dst.numSteps = src.num_steps
    dst.robustness = src.robustness

    if src.static_loss_type == "Euclidean":
        dst.staticLossType = StaticLossType.Euclidean
    elif src.static_loss_type == "ReproDisparity":
        dst.staticLossType = StaticLossType.ReproDisparity
    elif src.static_loss_type == "ReproDepthRatio":
        dst.staticLossType = StaticLossType.ReproDepthRatio
    elif src.static_loss_type == "ReproLogDepth":
        dst.staticLossType = StaticLossType.ReproLogDepth
    else:
        raise RuntimeError("Invalid static loss type specified.")

    dst.staticSpatialWeight = src.static_spatial_weight
    dst.staticDepthWeight = src.static_depth_weight

    if src.smooth_loss_type == "EuclideanLaplacian":
        dst.smoothLossType = SmoothLossType.EuclideanLaplacian
    elif src.smooth_loss_type == "ReproDisparityLaplacian":
        dst.smoothLossType = SmoothLossType.ReproDisparityLaplacian
    elif src.smooth_loss_type == "ReproDepthRatioConsistency":
        dst.smoothLossType = SmoothLossType.ReproDepthRatioConsistency
    elif src.smooth_loss_type == "ReproLogDepthConsistency":
        dst.smoothLossType = SmoothLossType.ReproLogDepthConsistency
    else:
        raise RuntimeError("Invalid smooth loss type specified.")

    dst.smoothStaticWeight = src.smooth_static_weight
    dst.smoothDynamicWeight = src.smooth_dynamic_weight
    dst.positionReg = src.position_regularization
    dst.scaleReg = src.scale_regularization
    dst.scaleRegGridSize = src.scale_regularization_grid_size
    dst.depthDeformRegInitial = src.deformation_regularization_initial
    dst.depthDeformRegFinal = src.deformation_regularization_final
    dst.adaptiveDeformationCost = src.adaptive_deformation_cost
    dst.spatialDeformReg = src.spatial_deformation_regularization
    dst.graduateDepthDeformReg = src.graduate_deformation_regularization
    dst.focalReg = src.focal_regularization

    dst.coarseToFine = src.coarse_to_fine
    dst.ctfLong = src.ctf_long
    dst.ctfShort = src.ctf_short

    dst.deferredSpatialOpt = src.deferred_spatial_opt
    dst.dsoLong = src.dso_long
    dst.dsoShort = src.dso_short

    dst.focalLong = src.focal_long

    if src.intr_opt == "Fixed":
        dst.intrOpt = IntrinsicsOptimization.Fixed
    elif src.intr_opt == "Shared":
        dst.intrOpt = IntrinsicsOptimization.Shared
    elif src.intr_opt == "PerFrame":
        dst.intrOpt = IntrinsicsOptimization.PerFrame
    else:
        raise RuntimeError("Invalid intrinsics optimization mode specified.")

    dst.fixPoses = src.fix_poses
    dst.fixDepthXforms = src.fix_depth_transforms
    dst.fixSpatialXforms = src.fix_spatial_transforms

    return dst


class PoseOptimizer:
    def __init__(
        self,
        base_dir: str,
        model_type: str,
        frames: List[int],
        opt_params: Nestedspace,
    ):
        self.base_dir = base_dir
        self.frames = frames

        # Initialize depth video with initial depth stream (unoptimized depth
        # estimate).
        self.depth_video = DepthVideo()
        discoverStreams = False
        DepthVideoImporter.importVideo(self.depth_video, base_dir, discoverStreams)
        self.depth_video.createColorStream("full", "color_full", ".png", CV_32FC3)
        self.depth_video.createColorStream("down", "color_down", ".raw", CV_32FC3)
        if os.path.isdir(os.path.join(base_dir, "dynamic_mask")):
            self.depth_video.createColorStream(
                "dynamic_mask", "dynamic_mask", ".png", CV_8UC1)

        # If a ground truth depth stream exists, we add it first, because we'll
        # allways be optimizing the last stream below.
        if os.path.exists(f"{base_dir}/depth_gt"):
            print("Importing ground truth depth...")
            self.depth_video.createDepthStream("depth_gt", "depth_gt", [-1, -1])
            poses_file = f"{base_dir}/depth_gt/poses.txt"
            if os.path.exists(poses_file):
                print("Importing ground truth poses...")
                gt_depth_stream = self.depth_video.numDepthStreams() - 1
                DepthVideoImporter.importPoses(
                    self.depth_video, poses_file, gt_depth_stream)

        # Import COLMAP poses and depth stream if exists.
        if os.path.exists(os.path.join(base_dir, "colmap_dense")) and os.path.exists(
            os.path.join(base_dir, "depth_colmap_dense")
        ):
            DepthVideoImporter.importColmapDepth(self.depth_video)
            self.depth_video.createDepthStream(
                "colmap_dense", "depth_colmap_dense_imported", [-1, -1]
            )
            colmap_file = os.path.join(base_dir, "colmap_dense/metadata.npz")
            stream = self.depth_video.depthStreamIndex("colmap_dense")
            DepthVideoImporter.importColmapRecon(
                self.depth_video, colmap_file, stream, False
            )

        # Add the estimated depth stream.
        depth_tag = f"depth_{model_type}"
        self.depth_video.createDepthStream(depth_tag, depth_tag, [-1, -1])

        # If using COLMAP, copy the poses to the Midas stream.
        if os.path.exists(os.path.join(base_dir, "colmap_dense")) and os.path.exists(
            os.path.join(base_dir, "depth_colmap_dense")
        ):
            src_ds_id = self.depth_video.depthStreamIndex("colmap_dense")
            dst_ds_id = self.depth_video.depthStreamIndex(depth_tag)
            self.copy_poses(src_ds_id, dst_ds_id)

        self.depth_video.printInfo()
        self.depth_video.save()

        self.opt_params = convert_opt_params(opt_params)
        self.use_global_scale = opt_params.use_global_scale

        # Initialize flow constraints
        flow_constraints_params = FlowConstraintsParams()
        # Remove out of bounds frames by explicitly setting clip=True.
        flow_constraints_params.frameRange.resolve(self.depth_video.numFrames(), True)
        self.flow_constraints = FlowConstraintsCollection(
            self.depth_video, flow_constraints_params)
        if opt_params.dynamic_constraints == "Mask":
            minDynamicDistance = 8
            self.flow_constraints.setStaticFlagFromDynamicMask(minDynamicDistance)
        elif opt_params.dynamic_constraints == "Ransac":
            self.flow_constraints.setStaticFlagFromRansac(opt_params.epipolar_dist_thresh)
        self.flow_constraints.save()

    def optimize_poses(self):
        frames_string = ",".join(str(x) for x in self.frames)

        # The underlying depth maps have been changed by fine-tuning, so we need
        # to clear our caches here.
        self.depth_video.clearDepthCaches()

        processor = DepthVideoProcessor(self.depth_video)
        params = DepthVideoProcessor.Params()

        # We're always optimizing the last depth stream.
        params.depthStream = self.depth_video.numDepthStreams() - 1

        params.frameRange.fromString(frames_string)

        params.poseOptimizer = self.opt_params
        params.poseOptimizer.frameRange.fromString(frames_string)

        # Reset all transforms, so we can optimize from scratch.

        params.op = DepthVideoProcessor.Op.ResetDepthXforms
        params.depthXformDesc.type = XformType.Depth
        params.depthXformDesc.depthType = DepthXformType.Global
        params.depthXformDesc.valueXform = ValueXformType.Scale
        processor.process(params)

        params.op = DepthVideoProcessor.Op.ResetSpatialXforms
        params.spatialXformDesc.type = XformType.Spatial
        params.spatialXformDesc.spatialType = SpatialXformType.Identity
        params.spatialXformDesc.valueXform = ValueXformType.Scale
        processor.process(params)

        processor.normalizeDepth(params, self.flow_constraints)

        # Now optimize poses and depth transforms jointly.
        processor.optimizePoses(params, self.flow_constraints)

        # Fixing the estimated pose and updating the depth xform to per-frame scaling
        if self.use_global_scale:
            params.poseOptimizer.fixPoses = True
            params.poseOptimizer.numSteps = 1
            params.poseOptimizer.coarseToFine = False

            # Reset depth transform
            params.op = DepthVideoProcessor.Op.ResetDepthXforms
            params.depthXformDesc.type = XformType.Depth
            params.depthXformDesc.depthType = DepthXformType.Global
            params.depthXformDesc.valueXform = ValueXformType.Scale
            processor.process(params)

            # Reset spatial transform
            params.op = DepthVideoProcessor.Op.ResetSpatialXforms
            params.spatialXformDesc.type = XformType.Spatial
            params.spatialXformDesc.spatialType = SpatialXformType.Identity
            params.spatialXformDesc.valueXform = ValueXformType.Scale
            processor.process(params)

            # Normalize depth
            processor.normalizeDepth(params, self.flow_constraints)

            # Optimize depth transfomration (while keeping the pose fixed)
            processor.optimizePoses(params, self.flow_constraints)

        self.depth_video.save()

    def copy_poses(self, src_ds_id, dst_ds_id):
        print(f"Copying poses for depth stream {src_ds_id} -> {dst_ds_id}...")
        src_ds = self.depth_video.depthStream(src_ds_id)
        dst_ds = self.depth_video.depthStream(dst_ds_id)

        dst_ds.resetDepthXforms(src_ds.depthXformDesc())
        dst_ds.resetSpatialXforms(src_ds.spatialXformDesc())

        for i in range(self.depth_video.numFrames()):
            src_f = src_ds.frame(i)
            dst_f = dst_ds.frame(i)

            # Update depth and spatial transformation
            dst_f.depthXform().copyFrom(src_f.depthXform())
            dst_f.spatialXform().copyFrom(src_f.spatialXform())

            # Update intrinsics and extrinsics
            dst_f.extrinsics = src_f.extrinsics
            dst_f.intrinsics = src_f.intrinsics

    # Make a copy of the last depth stream.
    def duplicate_last_depth_stream(self, name, dir):
        dst_ds_id = self.depth_video.numDepthStreams()
        src_ds_id = dst_ds_id - 1
        print(f"Copying depth stream {src_ds_id} -> {dst_ds_id}...")

        src_ds = self.depth_video.depthStream(src_ds_id)
        width = src_ds.width()
        height = src_ds.height()

        rel_dir = os.path.relpath(dir, self.base_dir)
        self.depth_video.createDepthStream(name, rel_dir, [width, height])
        print(f"Created depth stream '{name}' (dir '{rel_dir}').")

        # Copy the initialized poses to the other depth stream.
        self.copy_poses(src_ds_id, dst_ds_id)

        dst_ds = self.depth_video.depthStream(dst_ds_id)

        # Copy the actual depth maps.
        src_depth_dir = os.path.join(src_ds.path(), "depth")
        dst_depth_dir = os.path.join(dst_ds.path(), "depth")
        os.makedirs(dst_depth_dir, exist_ok=True)
        for i in self.frames:
            shutil.copyfile(
                f"{src_depth_dir}/frame_{i:06d}.raw",
                f"{dst_depth_dir}/frame_{i:06d}.raw")

        self.depth_video.save()

    def filter_depth(self, radius):
        dst_ds_id = self.depth_video.numDepthStreams()
        src_ds_id = dst_ds_id - 1
        print(f"Filtering depth stream {src_ds_id} -> {dst_ds_id}...")

        src_ds = self.depth_video.depthStream(src_ds_id)
        width = src_ds.width()
        height = src_ds.height()

        name = src_ds.name() + "_filtered"
        dir = f"{src_ds.path()}/{name}"
        rel_dir = os.path.relpath(dir, self.base_dir)
        self.depth_video.createDepthStream(name, rel_dir, [width, height])
        print(f"Created depth stream '{name}' (dir '{rel_dir}').")

        processor = DepthVideoProcessor(self.depth_video)
        params = DepthVideoProcessor.Params()

        frames_string = ",".join(str(x) for x in self.frames)
        params.frameRange.fromString(frames_string)

        print("Copying stream data...")
        params.op = DepthVideoProcessor.Op.Copy
        params.sourceDepthStream= src_ds_id
        params.depthStream = dst_ds_id
        processor.process(params)

        print("Filtering...")
        params.op = DepthVideoProcessor.Op.FlowGuidedFilter
        params.frameRadius = radius
        processor.process(params)

        print("Saving...")
        self.depth_video.saveDepth(dst_ds_id)
        self.depth_video.save()
