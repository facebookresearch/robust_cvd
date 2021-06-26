#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Suppress several verbose warnings for easier debugging.
import warnings  # isort:skip

# warnings.simplefilter("ignore", ResourceWarning)
# warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore")

import itertools
import json
import math
import os
import os.path as osp
import time
from os.path import join as pjoin
from typing import Dict

import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monodepth.depth_model_registry import get_depth_model

import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset
from loss.joint_loss import JointLoss
from loss.loss_params import LossParams
from pose_optimization import PoseOptimizer
from utils import image_io, visualization
from utils.helpers import SuppressedStdout
from utils.torch_helpers import to_device


def get_tensorboard_prompt(log_dir: str) -> str:
    """
    return a prompt string leading to tensorboard gui
    """
    return (
        "\n======================\n"
        "View tensorboard by running command:\n"
        f"tensorboard --port=8098 --logdir {log_dir}\n"
        "then go to the prompted link\n"
        "======================\n"
    )


class DepthFineTuningParams:
    """Options about finetune parameters."""

    @staticmethod
    def add_arguments(parser):
        parser = LossParams.add_arguments(parser)

        parser.add_argument(
            "--optimizer",
            default="Adam",
            choices=optimizer.OPTIMIZER_NAMES,
            help="optimizer to train the network",
        )
        parser.add_argument(
            "--val_epoch_freq",
            type=int,
            default=-1,
            help="Validation epoch frequency. Set to -1 to disable validation.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0,
            help="Learning rate for the training. If <= 0 it will be set"
            " automatically to the default for the specified model adapter.",
        )
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--num_epochs", type=int, default=10)
        parser.add_argument("--pose_opt_freq", type=int, default=1)

        parser.add_argument("--log_dir", help="folder to log tensorboard summary")

        parser.add_argument(
            "--display_freq",
            type=int,
            default=100,
            help="frequency of showing training results on screen",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=1,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=1,
            help="frequency of saving checkpoints at the end of epochs",
        )

        parser.add_argument("--save_eval_images", action="store_true")
        parser.add_argument("--save_depth_xform_maps", action="store_true")
        parser.add_argument("--save_checkpoints", action="store_true")
        parser.add_argument("--save_tensorboard", action="store_false")
        parser.add_argument("--tensorboard_log_path", type=str)
        parser.add_argument("--save_scene_flow_vis", action="store_true")
        parser.add_argument(
            "--save_intermediate_depth_streams_freq",
            type=int,
            default=0,
            help="to not store if 0, and store with freq >0",
        )
        parser.add_argument("--save_depth_visualization", action="store_true")

        return parser


def log_loss_stats(
    writer: SummaryWriter,
    name_prefix: str,
    loss_meta: Dict[str, torch.Tensor],
    n: int,
    log_histogram: bool = False,
):
    """
    loss_meta: sub_loss_name: individual losses
    """
    for sub_loss_name, loss_value in loss_meta.items():
        sub_loss_full_name = name_prefix + "/" + sub_loss_name

        writer.add_scalar(
            sub_loss_full_name + "/max",
            loss_value.max(),
            n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/min",
            loss_value.min(),
            n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/mean",
            loss_value.mean(),
            n,
        )

        if log_histogram:
            writer.add_histogram(sub_loss_full_name, loss_value, n)


def write_summary(writer, mode_name, input_images, depth, metadata, n_iter):
    DIM = -3
    B = depth.shape[0]

    inv_depth_pred = depth.unsqueeze(DIM)

    mask = torch.stack(metadata["geometry_consistency"]["masks"], dim=1)

    def to_vis(x):
        return x[:8].transpose(0, 1).reshape((-1,) + x.shape[DIM:])

    writer.add_image(
        mode_name + "/image",
        vutils.make_grid(to_vis(input_images), nrow=B, normalize=True),
        n_iter,
    )
    writer.add_image(
        mode_name + "/pred_full",
        vutils.make_grid(to_vis(1.0 / inv_depth_pred), nrow=B, normalize=True),
        n_iter,
    )
    writer.add_image(
        mode_name + "/mask",
        vutils.make_grid(to_vis(mask), nrow=B, normalize=True),
        n_iter,
    )


def log_loss(
    writer: SummaryWriter,
    mode_name: str,
    loss: torch.Tensor,
    loss_meta: Dict[str, torch.Tensor],
    niters: int,
):
    main_loss_name = mode_name + "/loss"

    writer.add_scalar(main_loss_name, loss, niters)
    log_loss_stats(writer, main_loss_name, loss_meta, niters)


def make_tag(params, exp_tag="short"):
    if exp_tag == "short":
        return LossParams.make_str(params, exp_tag)
    else:
        return (
            LossParams.make_str(params, exp_tag)
            + f"_LR{params.learning_rate}"
            + f"_BS{params.batch_size}"
            + f"_O{params.optimizer.lower()}"
            + f"_S{params.scaling}"
        )


class DepthFineTuner:
    def __init__(self, range_dir, frames, base_dir, params):
        self.frames = frames
        self.params = params
        self.base_dir = base_dir
        self.range_dir = range_dir
        self.out_dir = pjoin(self.range_dir, make_tag(params, params.exp_tag))
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Fine-tuning directory: '{self.out_dir}'")

        # Checkpoint directory
        if self.params.save_checkpoints:
            self.checkpoints_dir = pjoin(self.out_dir, "checkpoints")
            os.makedirs(self.checkpoints_dir, exist_ok=True)

        model = get_depth_model(params.model_type)
        self.model = model()

        self.reference_disparity = {}

    def save_depth(self, dir: str = None, frames=None):
        save_depth_start_time = time.perf_counter()

        if dir is None:
            dir = self.depth_dir

        if frames is None:
            frames = self.frames

        color_fmt = pjoin(self.base_dir, "color_down", "frame_{:06d}.raw")
        depth_dir = pjoin(dir, "depth")
        depth_fmt = pjoin(depth_dir, "frame_{:06d}")

        print(f"Saving depth to '{dir}'...")
        print(f"Current 'frames':\n {frames}")

        num_gpus = torch.cuda.device_count()
        batch_size = num_gpus
        dataset = VideoFrameDataset(color_fmt, frames)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model.eval()

        os.makedirs(depth_dir, exist_ok=True)
        for data in data_loader:

            # Skip this batch if all frames are out of range.
            if frames is not None and not any(i in frames for i in data[1]["frame_id"]):
                continue

            data = to_device(data)
            stacked_images, metadata = data

            depth_batch = self.model.forward(stacked_images, metadata)
            batch_size, _, _ = depth_batch.shape

            depth_batch = depth_batch.detach().cpu().numpy().squeeze()
            inv_depth_batch = 1.0 / depth_batch

            for i in range(batch_size):
                if batch_size > 1:
                    inv_depth = inv_depth_batch[i, :, :].squeeze()
                else:
                    inv_depth = inv_depth_batch
                frame_id = metadata["frame_id"][i]

                if frames is None or frame_id in frames:
                    image_io.save_raw_float32_image(
                        depth_fmt.format(frame_id) + ".raw", inv_depth
                    )

        if self.params.save_depth_visualization:
            print(f"Visualizing depth dir: {depth_dir}...")
            with SuppressedStdout():
                visualization.visualize_depth_dir(
                    src_dir=depth_dir, dst_dir=depth_dir, force=True
                )

        save_depth_end_time = time.perf_counter()
        save_depth_duration = save_depth_end_time - save_depth_start_time
        print(
            f"Complete saving depth for pose optimization in {save_depth_duration:.2f}s"
        )

    def load_reference_disparity(self, frame):
        if frame not in self.reference_disparity:
            filename = pjoin(
                self.params.path,
                "depth_colmap_dense",
                "depth",
                "frame_{:06d}.raw".format(frame),
            )
            disparity = image_io.load_raw_float32_image(filename)
            disparity = cv2.resize(
                disparity, (224, 384), interpolation=cv2.INTER_NEAREST
            )
            self.reference_disparity[frame] = disparity
        return self.reference_disparity[frame]

    def fine_tune(self, writer=None):
        meta_file = None

        if self.params.recon == "colmap":
            if self.params.scaling == "extrinsics":
                meta_file = pjoin(self.range_dir, "metadata_scaled.npz")
            else:
                meta_file = pjoin(self.base_dir, "colmap_dense", "metadata.npz")

        print("Start depth finetuning...")

        use_temporal_smooth_loss = (
            self.params.lambda_smooth_disparity > 0
            or self.params.lambda_smooth_reprojection > 0
            or self.params.lambda_smooth_depth_ratio > 0
        )

        dataset = VideoDataset(
            self.base_dir,
            self.frames,
            self.params.min_mask_ratio,
            use_temporal_smooth_loss,
            meta_file,
            self.params.recon,
        )
        train_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        # Even if we're using the COLMAP pipeline, we're initializing the pose
        # optimizer here, because it will create a depth video container for us.
        pose_optimizer = PoseOptimizer(
            self.base_dir, self.params.model_type, self.frames, self.params.opt
        )

        if self.params.recon == "i3d":
            pose_optimizer.optimize_poses()

        if self.params.save_intermediate_depth_streams_freq > 0:
            self.depth_dir = os.path.join(self.out_dir, "depth_e0000")
            pose_optimizer.duplicate_last_depth_stream("e0000", self.depth_dir)
        else:
            self.depth_dir = self.out_dir
            pose_optimizer.duplicate_last_depth_stream("fine_tuned", self.depth_dir)

        if self.params.recon == "i3d":
            dataset.update_poses(pose_optimizer.depth_video)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Only enable back-propagation for the PCA realted parameters if specified.
        if self.params.model_type == "midas2_pca":
            for name, param in self.model.named_parameters():
                if name == "model.scale_params" or name == "model.shift_params":
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Only cover tunable PCA parameters in loss computation.
        criterion = JointLoss(
            self.params, parameters_init=[p.clone() for p in self.model.parameters()]
        )

        if self.params.save_tensorboard and writer is None:
            if self.params.tensorboard_log_path:
                log_dir = self.params.tensorboard_log_path
            else:
                log_dir = pjoin(self.out_dir, "tensorboard")

            # Print the prompt to view the tensorboard.
            print(get_tensorboard_prompt(log_dir))
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

        # Only include tunable PCA parameters in the optimizer if specified.
        if self.params.model_type == "midas2_pca":
            opt = optimizer.create(
                self.params.optimizer,
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.params.learning_rate,
                betas=(0.9, 0.999),
            )
        else:
            opt = optimizer.create(
                self.params.optimizer,
                self.model.parameters(),
                self.params.learning_rate,
                betas=(0.9, 0.999),
            )

        self.model.train()

        def validate(epoch, niters):
            val_start_time = time.perf_counter()

            loss_meta = self.eval_and_save(criterion, val_data_loader, epoch, niters)
            if writer is not None:
                log_loss_stats(
                    writer, "validation", loss_meta, epoch, log_histogram=True
                )

            val_end_time = time.perf_counter()
            val_duration = val_end_time - val_start_time

            print(
                f"Complete Validation for epoch {epoch} ({niters} iterations) in {val_duration:.2f}s."
            )

        if self.params.val_epoch_freq >= 0:
            validate(epoch=0, niters=0)

        # Disable inplace relu for batch-wise PCA modulation
        def disable_relu_inplace(model) -> None:
            for child_name, child in model.named_children():
                if isinstance(child, torch.nn.ReLU):
                    setattr(model, child_name, torch.nn.ReLU(inplace=False))
                else:
                    disable_relu_inplace(child)

        # Retrieve intially computed depth predictions for loss computation.
        # depth_fmt = "frame_{:06d}.raw"
        initial_depth_dir = osp.join(self.base_dir, f"depth_{self.params.model_type}", "depth")

        depth_names = [
            n for n in os.listdir(initial_depth_dir) if os.path.splitext(n)[-1] == ".raw"
        ]
        depth_names = sorted(depth_names)

        all_depth_orig = {}
        for depth_name in depth_names:
            depth_path = osp.join(initial_depth_dir, depth_name)
            depth_orig = 1.0 / image_io.load_raw_float32_image(depth_path)
            all_depth_orig[depth_name] = torch.from_numpy(depth_orig)

        def retrieve_depth_orig(metadata) -> torch.Tensor:
            """
            Retrieve the corresponding original depths for loss computation.
            """
            indices = metadata["geometry_consistency"]["indices"]
            indices_list = indices.cpu().numpy().tolist()
            indices_list = list(itertools.chain(*indices_list))

            depth_orig = []
            for idx in indices_list:
                depth_orig.append(all_depth_orig.get(f"frame_{idx:06d}.raw"))

            self.depth_orig = torch.stack(depth_orig)

            return self.depth_orig

        # Training loop.
        total_iters = 0
        for epoch in range(self.params.num_epochs):
            epoch_start_time = time.perf_counter()

            for data in train_data_loader:

                if self.params.model_type == "midas2_pca":
                    print(f"'scale_params': {self.model.model.scale_params}")
                    print(f"'shift_params': {self.model.model.shift_params}")

                iter_start_time = time.perf_counter()

                data = to_device(data)
                stacked_img, metadata = data
                print(f"Size of stacked_img: {stacked_img.shape}")

                print(f"Current batch_size: {self.params.batch_size}")
                depth = self.model(stacked_img, metadata)

                # Apply per-frame scales
                if self.params.recon == "colmap" and self.params.scaling == "depth":
                    indices = metadata["geometry_consistency"]["indices"]
                    scale = torch.Tensor(
                        indices.shape[0], indices.shape[1], 1, 1
                    ).cuda()
                    for pair in range(indices.shape[0]):
                        for i in range(2):
                            frame = int(indices[pair][i])
                            ref_disp = self.load_reference_disparity(frame)
                            valid = ~np.logical_or(
                                np.isinf(ref_disp), np.isnan(ref_disp)
                            )
                            est_disp = 1.0 / depth[pair, i, :].detach().cpu()
                            pixel_scales = (est_disp / ref_disp)[valid]
                            image_scale = np.median(pixel_scales)
                            scale[pair, i] = float(image_scale)
                            print(f"Frame {frame}: scale = {image_scale}.")
                    depth = depth * scale

                opt.zero_grad()

                # Retrieve original depth predictions for contrast loss computation.
                depth_orig = retrieve_depth_orig(metadata)
                _, h, w = depth_orig.shape
                # Reshape (x, h, w) to (b, n, h, w) to match depth.
                depth_orig = depth_orig.view(-1, 2, h, w)
                depth_orig = depth_orig.to(depth.device)

                # Loss computation.
                loss, loss_meta, _ = criterion(
                    stacked_img,
                    depth_orig,
                    depth,
                    metadata,
                    parameters=self.model.parameters(),
                )

                pairs = metadata["geometry_consistency"]["indices"]
                pairs = pairs.cpu().numpy().tolist()

                print(f"Epoch = {epoch}, pairs = {pairs}, loss = {loss[0]}")
                if torch.isnan(loss):
                    print("Loss is NaN. Skipping.")
                    continue

                loss.backward()
                opt.step()

                total_iters += stacked_img.shape[0]

                print(f"total_iters: {total_iters}")

                if writer is not None and total_iters % self.params.print_freq == 0:
                    log_loss(writer, "Train", loss, loss_meta, total_iters)

                if writer is not None and total_iters % self.params.display_freq == 0:
                    write_summary(
                        writer, "Train", stacked_img, depth, metadata, total_iters
                    )

                iter_end_time = time.perf_counter()
                iter_duration = iter_end_time - iter_start_time
                print(f"Iteration took {iter_duration:.2f}s.")

            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} took {epoch_duration:.2f}s.")

            if (
                self.params.val_epoch_freq >= 0
                and (epoch + 1) % self.params.val_epoch_freq == 0
            ):
                validate(epoch + 1, total_iters)

            if (
                self.params.save_checkpoints
                and (epoch + 1) % self.params.save_epoch_freq == 0
            ):
                file_name = pjoin(self.checkpoints_dir, f"{epoch + 1:04d}.pth")
                self.model.save(file_name)

            if (
                self.params.save_intermediate_depth_streams_freq > 0
                and (epoch + 1) % self.params.save_intermediate_depth_streams_freq == 0
            ):
                self.save_depth(frames=self.frames)

            if (
                self.params.recon == "i3d"
                and (epoch + 1) % self.params.pose_opt_freq == 0
            ):
                if self.params.save_intermediate_depth_streams_freq > 0:
                    # Create new depth stream for optimized poses.
                    epoch_str = f"e{epoch:04d}_opt"
                    self.depth_dir = os.path.join(self.out_dir, f"depth_{epoch_str}")
                    pose_optimizer.duplicate_last_depth_stream(
                        epoch_str, self.depth_dir
                    )

                # Pose optimization with depth/spatial deformation
                pose_opt_start_time = time.perf_counter()

                pose_optimizer.optimize_poses()
                dataset.update_poses(pose_optimizer.depth_video)

                pose_opt_end_time = time.perf_counter()
                pose_opt_duration = pose_opt_end_time - pose_opt_start_time

                print(f"Complete pose optimization in {pose_opt_duration:.2f}s")

                if (
                    self.params.save_intermediate_depth_streams_freq > 0
                    and (epoch + 1) % self.params.save_intermediate_depth_streams_freq
                    == 0
                ):
                    self.save_depth(frames=self.frames)

            if (
                self.params.save_intermediate_depth_streams_freq > 0
                and (epoch + 1) % self.params.save_intermediate_depth_streams_freq == 0
                and epoch + 1 < self.params.num_epochs
            ):
                # Create depth stream for the next epoch.
                epoch_str = f"e{epoch + 1:04d}"
                self.depth_dir = os.path.join(self.out_dir, f"depth_{epoch_str}")
                pose_optimizer.duplicate_last_depth_stream(epoch_str, self.depth_dir)

        # Validate the last epoch, unless it was just done in the loop above.
        if (
            self.params.val_epoch_freq >= 0
            and self.params.num_epochs % self.params.val_epoch_freq != 0
        ):
            validate(epoch=self.params.num_epochs, niters=total_iters)

        if self.params.post_filter:
            pose_optimizer.filter_depth(self.params.filter_radius)

        print("Finished Filtering.")

    def eval_and_save(
        self, criterion, data_loader, epoch, niters
    ) -> Dict[str, torch.Tensor]:
        """
        Note this function asssumes the structure of the data produced by data_loader
        """

        eval_dir = pjoin(self.out_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        N = len(data_loader.dataset)
        loss_dict = {}
        saved_frames_depth = set()
        saved_frames_xform = set()
        max_frame_index = 0
        all_pairs = []

        def suffix(epoch, niters):
            return "_e{:04d}_iter{:06d}".format(epoch, niters)

        def save_eval_depth(
            dispartiy_batch,
            batch_indices,
            out_dir,
            suf,
            is_last_epoch,
            is_zero_epoch,
            saved_frames_depth,
        ):
            dispartiy_max = disparity_batch.max()
            for disparities, indices in zip(disparity_batch, batch_indices):
                for disparity, index in zip(disparities, indices):
                    # Only save frames not saved before.
                    if index in saved_frames_depth:
                        continue
                    saved_frames_depth.add(index)

                    # Saving depth visualization at the current epoch
                    fn_pre = pjoin(out_dir, "eval", "depth_{:06d}{}".format(index, suf))
                    image_io.save_raw_float32_image(fn_pre + ".raw", disparity)

                    disparity_vis = visualization.visualize_depth(
                        disparity, depth_min=0, depth_max=dispartiy_max
                    )
                    cv2.imwrite(fn_pre + ".png", disparity_vis)

            return saved_frames_depth

        def save_eval_depth_xform(
            depth_xform_maps_batch, batch_indices, out_dir, suf, saved_frames_xform
        ):
            suf = suffix(epoch, niters)
            depth_xform_max = depth_xform_maps_batch.max()

            for depth_xform_maps, indices in zip(depth_xform_maps_batch, batch_indices):
                for depth_xform_map, index in zip(depth_xform_maps, indices):
                    # Only save frames not saved before.
                    if index in saved_frames_xform:
                        continue
                    saved_frames_xform.add(index)

                    fn_pre = pjoin(
                        self.out_dir, "eval", "scale_{:06d}{}".format(index, suf)
                    )
                    image_io.save_raw_float32_image(fn_pre + ".raw", depth_xform_map)

                    depth_xform_map = depth_xform_map / depth_xform_max
                    depth_xform_map_vis = np.uint8(depth_xform_map * 255)

                    cv2.imwrite(fn_pre + ".png", depth_xform_map_vis)

            return saved_frames_xform

        def get_scene_flow_idx(self, index_scene_flow, index_frames):
            if index_scene_flow == 0:
                return index_frames
            elif index_scene_flow == 1:
                return [index_frames[1], index_frames[0]]
            elif index_scene_flow == 2:
                return [index_frames[0], index_frames[0] + 1]
            elif index_scene_flow == 3:
                return [index_frames[0], index_frames[0] - 1]
            elif index_scene_flow == 4:
                return [index_frames[1], index_frames[1] + 1]
            elif index_scene_flow == 5:
                return [index_frames[1], index_frames[1] - 1]

        def save_scene_flow(scene_flow, batch_indices):
            scene_flow_vis = visualization.visualize_scene_flow(scene_flow)

            for index_scene_flow in range(len(scene_flow_vis)):
                scene_flow_vis_batch = scene_flow_vis[index_scene_flow]

                for scene_flow_vis_cur, indices in zip(
                    scene_flow_vis_batch, batch_indices
                ):
                    index_frame = get_scene_flow_idx(index_scene_flow, indices)
                    fn_pre = pjoin(
                        self.out_dir,
                        "eval",
                        "scene_flow_{:06d}_{:06d}_{}".format(
                            index_frame[0], index_frame[1], suf
                        ),
                    )
                    cv2.imwrite(fn_pre + ".png", scene_flow_vis_cur.transpose(1, 2, 0))

        is_zero_epoch = epoch == 0
        is_last_epoch = epoch == self.params.num_epochs

        # Looping over all the training batches in the dataloader
        for _, data in zip(range(N), data_loader):
            data = to_device(data)
            stacked_img, metadata = data

            with torch.no_grad():
                depth = self.model(stacked_img, metadata)

            # Apply per-frame scales for COLMAP
            if self.params.recon == "colmap" and self.params.scaling == "depth":
                indices = metadata["geometry_consistency"]["indices"]
                scale = torch.Tensor(indices.shape[0], indices.shape[1], 1, 1).cuda()
                for pair in range(indices.shape[0]):
                    for i in range(2):
                        frame = int(indices[pair][i])
                        ref_disp = self.load_reference_disparity(frame)
                        valid = ~np.logical_or(np.isinf(ref_disp), np.isnan(ref_disp))
                        est_disp = 1.0 / depth[pair, i, :].detach().cpu()
                        pixel_scales = (est_disp / ref_disp)[valid]
                        image_scale = np.median(pixel_scales)
                        scale[pair, i] = float(image_scale)
                        print(f"Frame {frame}: scale = {image_scale}.")
                depth = depth * scale

            batch_indices = (
                metadata["geometry_consistency"]["indices"].cpu().numpy().tolist()
            )

            # Update the maximum frame index and pairs list.
            max_frame_index = max(max_frame_index, max(itertools.chain(*batch_indices)))
            all_pairs += batch_indices

            # Compute and store losses.
            _, loss_meta, scene_flow = criterion(
                stacked_img,
                depth,
                metadata,
                parameters=self.model.parameters(),
            )

            for loss_name, losses in loss_meta.items():
                if loss_name not in loss_dict:
                    loss_dict[loss_name] = {}

                for indices, loss in zip(batch_indices, losses):
                    loss_dict[loss_name][str(indices)] = loss.item()

            suf = suffix(epoch, niters)

            # Save the current depth predictions
            if self.params.save_eval_images or is_last_epoch or is_zero_epoch:
                disparity_batch = 1.0 / depth.cpu().detach().numpy()
                saved_frames_depth = save_eval_depth(
                    disparity_batch,
                    batch_indices,
                    self.out_dir,
                    suf,
                    is_last_epoch,
                    is_zero_epoch,
                    saved_frames_depth,
                )

            # Save the current depth transformation
            if self.params.save_depth_xform_maps:
                depth_xform_maps_batch = metadata["scales"].cpu().detach().numpy()
                saved_frames_xform = save_eval_depth_xform(
                    depth_xform_maps_batch,
                    batch_indices,
                    self.out_dir,
                    suf,
                    saved_frames_xform,
                )

            # Save scene flow visualization
            if self.params.save_scene_flow_vis and scene_flow is not None:
                save_scene_flow(scene_flow, batch_indices)

        loss_meta = {
            loss_name: torch.tensor(tuple(loss.values()))
            for loss_name, loss in loss_dict.items()
        }
        loss_dict["mean"] = {k: v.mean().item() for k, v in loss_meta.items()}

        with open(pjoin(self.out_dir, "eval", "loss{}.json".format(suf)), "w") as f:
            json.dump(loss_dict, f)

        # Print verbose summary to stdout.
        index_width = int(math.ceil(math.log10(max_frame_index)))
        loss_names = list(loss_dict.keys())
        loss_names.remove("mean")
        loss_format = {}

        for name in loss_names:
            max_value = max(loss_dict[name].values())
            width = math.ceil(math.log10(max_value + 1))
            loss_format[name] = f"{width+7}.6f"

        for pair in sorted(all_pairs):
            line = f"({pair[0]:{index_width}d}, {pair[1]:{index_width}d}): "
            line += ", ".join(
                [
                    f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
                    for name in loss_names
                ]
            )
            print(line)

        print(
            "Mean: "
            + " " * (2 * index_width)
            + ", ".join(
                [
                    f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
                    for name in loss_names
                ]
            )
        )

        return loss_meta
