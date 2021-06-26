#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn as nn
from loss import distance
from utils.torch_helpers import _device
from utils.geometry import (
    pixel_grid,
    focal_length,
    project,
    pixels_to_points,
    points_cam_to_world,
    world_to_points_cam,
    sample,
)
from utils.loss import (
    select_tensors,
    weighted_mean_loss,
)


class SceneFlowLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # self.robust_dist = distance.create(opt)
        self.robust_dist_static = distance.create(opt.distance_type_static, opt)
        self.robust_dist_smooth = distance.create(opt.distance_type_smooth, opt)

    def smooth_scene_flow_loss(self, points_cam, pixels, intrinsics_pair, extrinsics, flows_n, masks_n, valid_n):
        """ Static scene flow loss

        Args:
            N = 6: sampled triplets
            - ref, trg, ref - 1, ref + 1, trg - 1, trg + 1
            points_cam (B, 3, H, W) * N: points in local camera coordinate.
            pixels (B, 2, H, W) * N: pixel cooredindate
            intrinsics_pair (2, B, 4)
            extrinsics (B, 3, 4) * N:  # extrinsics of each frame. Each (3, 4) = [R, t]
            flows_n (B, 2, H, W) * 4: backward/forward flow. ref - 1, ref + 1, trg - 1, trg + 1
            masks_n (B, 1, H, W) * 4: flow masks
            valid_n (B, 2, 1, 1):  indicator of whether a flow neighbor is a valid frame
        """
        smooth_reproj_losses = []
        smooth_disparity_losses = []
        smooth_depth_ratio_losses = []

        scene_flow_neighbor = []

        pair_idx = [0, 1]
        neighbor_idx = [2, 3, 4, 5]

        # Reference and target
        points_cam_pair = points_cam[pair_idx]
        pixels_pair = pixels[pair_idx]
        extrinsics_pair = extrinsics[pair_idx]

        # Temporal neighbors of reference and target frames
        points_cam_n = points_cam[neighbor_idx]
        extrinsics_n = extrinsics[neighbor_idx]

        for anchor_idx in pair_idx:
            points_cam_ref = points_cam_pair[anchor_idx]              # (B, 3, H, W)
            pixels_ref = pixels_pair[anchor_idx]                      # (B, 2, H, W)
            extrinsics_ref = extrinsics_pair[anchor_idx]              # (B, 3, 4)
            intrinsics_ref = intrinsics_pair[anchor_idx]              # (B, 4)
            valid_n_ref = valid_n[:, anchor_idx, :, :].unsqueeze(-1)  # (B, 1, 1, ,1)

            # Convert points reference camera coordindate to world coordindate
            points_world_ref = points_cam_to_world(points_cam_ref, extrinsics_ref)

            bw_idx, fw_idx = anchor_idx * 2, anchor_idx * 2 + 1

            # Find matched points (B, 2, H, W) using forward/backward optical flow
            matched_pixels_bw = pixels_ref + flows_n[bw_idx]
            matched_pixels_fw = pixels_ref + flows_n[fw_idx]

            # Sample the matched points in target camera coordidate
            points_cam_trg_bw = sample(points_cam_n[bw_idx], matched_pixels_bw)
            points_cam_trg_fw = sample(points_cam_n[fw_idx], matched_pixels_fw)

            # Convert the matched points from target camera to world coordindate
            points_world_trg_bw = points_cam_to_world(points_cam_trg_bw, extrinsics_n[bw_idx])
            points_world_trg_fw = points_cam_to_world(points_cam_trg_fw, extrinsics_n[fw_idx])

            scene_flow_fw = points_world_trg_fw - points_world_ref
            scene_flow_bw = points_world_trg_bw - points_world_ref

            # The forward scene flow and backward scene flow should cancel out. We minimize the residual.
            scene_flow_residual = scene_flow_fw + scene_flow_bw
            points_world_ref_s = points_world_ref + scene_flow_residual

            points_cam_ref_s = world_to_points_cam(points_world_ref_s, extrinsics_ref)

            # mask out invalid matches
            mask = valid_n_ref * masks_n[bw_idx] * masks_n[fw_idx]    # (B, 1, H, W)

            # === Reprojection loss ===
            if self.opt.lambda_smooth_reprojection > 0:
                pixels_ref_s = project(points_cam_ref_s, intrinsics_ref)
                reproj_dist = torch.norm(pixels_ref_s - pixels_ref, dim=1, keepdim=True)

                smooth_reproj_losses.append(
                    weighted_mean_loss(self.robust_dist_smooth(reproj_dist), mask)
                )

            # === Disparity loss ===
            if self.opt.lambda_smooth_disparity > 0:
                f = torch.mean(focal_length(intrinsics_ref))
                disp_diff = 1.0 / points_cam_ref_s[:, -1:, ...] \
                    - 1.0 / points_cam_ref[:, -1:, ...]
                smooth_disparity_losses.append(
                    f * weighted_mean_loss(self.robust_dist_smooth(disp_diff), mask)
                )

            # === Depth ratio loss ===
            if self.opt.lambda_smooth_depth_ratio > 0:
                # camera is facing the -z axis
                depth_ref = torch.abs(points_cam_ref[:, -1:, ...])
                depth_ref_s = torch.abs(points_cam_ref_s[:, -1:, ...])

                # compute the min and max values for both depth values
                depth_min = torch.min(depth_ref, depth_ref_s)  # (B, 1, H, W)
                depth_max = torch.max(depth_ref, depth_ref_s)  # (B, 1, H, W)

                # compute the depth ratio. Pre-multiply weights before applying robust functions
                depth_ratio = self.opt.lambda_smooth_depth_ratio * torch.log(depth_min / depth_max)

                smooth_depth_ratio_losses.append(
                    weighted_mean_loss(self.robust_dist_smooth(depth_ratio), mask)
                )

            # Remove inconsistent scene flow for visualization
            scene_flow_fw = mask * scene_flow_fw
            scene_flow_bw = mask * scene_flow_bw

            scene_flow_neighbor.append(scene_flow_fw.cpu().detach().numpy())
            scene_flow_neighbor.append(scene_flow_bw.cpu().detach().numpy())

        return smooth_reproj_losses, smooth_disparity_losses, smooth_depth_ratio_losses, scene_flow_neighbor

    def static_scene_flow_loss(self, points_cam, pixels, extrinsics, flows, masks):
        """ Static scene flow loss

        Args:
            N = 2: sampled pair
            points_cam (N, B, 3, H, W): points in local camera coordinate.
            pixels (N, B, 2, H, W):
            extrinsics (N, B, 3, 4)
            flows (N, B, 2, H, W)
            masks (N, B, 1, H, W)
        """
        inv_idxs = [1, 0]

        static_losses = []
        scene_flow_pair = []

        for (
            points_cam_ref,
            points_cam_trg,
            pixels_ref,
            flows_ref,
            masks_ref,
            extrinsics_ref,
            extrinsics_tgt,
        ) in zip(
            points_cam,
            points_cam[inv_idxs],
            pixels,
            flows,
            masks,
            extrinsics,
            extrinsics[inv_idxs],
        ):
            # Convert points reference camera coordindate to world coordindate
            points_world_ref = points_cam_to_world(points_cam_ref, extrinsics_ref)

            # Find matched points (B, 2, H, W) using optical flow
            matched_pixels_tgt = pixels_ref + flows_ref

            # Sample the matched points in target camera coordidate
            points_cam_trg = sample(points_cam_trg, matched_pixels_tgt)

            # Convert the matched points from target camera to world coordindate
            points_world_trg = points_cam_to_world(points_cam_trg, extrinsics_tgt)

            # Compute static scene flow loss
            scene_flow = points_world_ref - points_world_trg
            points_dist = torch.norm(scene_flow, dim=1, keepdim=True)

            weight_map = masks_ref * torch.abs(1.0 / points_cam_ref[:, -1:, ...])
            static_losses.append(
                weighted_mean_loss(self.robust_dist_static(points_dist), weight_map)
            )

            # Remove inconsistent scene flow
            scene_flow = weight_map * scene_flow
            scene_flow_pair.append(scene_flow.cpu().detach().numpy())

        return static_losses, scene_flow_pair

    def scene_flow_loss(self, points_cam, metadata, pixels):
        """Scene Flow Loss.

        The scene flow loss consists of two parts:
            - static scene flow loss: geometric consistency loss
            - temporal smoothness scene flow loss
        Both losses are measured in 3D world coordindates.

        Using only static loss: N = 2
        Both losses are measured in 3D world coordindates.

        Args:
            points_cam (B, N, 3, H, W): points in local camera coordinate.
            pixels (B, N, 2, H, W)
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include entries as below.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4), # (fx, fy, cx, cy)
                    'geometry_consistency':
                        {
                            'flows':    (B, 2, H, W),) * 2 in pixels.
                                        For k in range(2) (ref or tgt),
                                            pixel p = pixels[indices[b, k]][:, i, j]
                                        correspond to
                                            p + flows[k][b, :, i, j]
                                        in frame indices[b, (k + 1) % 2].
                            'masks':    ((B, 1, H, W),) * 2. Masks of valid flow
                                        matches. Values are 0 or 1.
                        }
                    'temporal_smoothness': (if using temporal smoothness scene flow loss)
                        {
                            'indices':  torch.tensor (4),
                                indices for consecutive consecutive frames
                                        [(ref_index-1, ref_index + 1, tgt_index - 1, tgt_index + 1), ...]
                            'flows':    ((B, 2, H, W),)* 4 in pixels.
                                flows[0][b,:, i, j] - flow map for ref_index -> ref_index - 1 (backward flow)
                                flows[1][b,:, i, j] - flow map for ref_index -> ref_index + 1 (forward flow)
                                flows[2][b,:, i, j] - flow map for tgt_index -> tgt_index - 1 (backward flow)
                                flows[3][b,:, i, j] - flow map for tgt_index -> tgt_index + 1 (forward flow)
                            'masks':    ((B, 1, H, W),) * 4. Masks of valid flow matches
                                to compute the consistency in training.
                                Values are 0 or 1.
                    }
                }
        """
        extrinsics = metadata["extrinsics"]
        extrinsics = select_tensors(extrinsics)
        intrinsics = metadata["intrinsics"]
        intrinsics = select_tensors(intrinsics)
        points_cam = select_tensors(points_cam)
        pixels = select_tensors(pixels)

        pair_idx = [0, 1]

        points_cam_pair = points_cam[pair_idx]
        pixels_pair = pixels[pair_idx]
        extrinsics_pair = extrinsics[pair_idx]
        intrinsics_pair = intrinsics[pair_idx]

        geom_meta = metadata["geometry_consistency"]
        flows_pair = (flows for flows in geom_meta["flows"])
        masks_pair = (masks for masks in geom_meta["masks"])

        static_losses, smooth_reproj_losses, smooth_disparity_losses = [], [], []
        scene_flow_pair, scene_flow_neighbor = [], []

        if self.opt.lambda_scene_flow_static > 0:
            static_losses, scene_flow_pair = self.static_scene_flow_loss(
                points_cam_pair,
                pixels_pair,
                extrinsics_pair,
                flows_pair,
                masks_pair,
            )

        if (
            self.opt.lambda_smooth_disparity > 0
            or self.opt.lambda_smooth_reprojection > 0
            or self.opt.lambda_smooth_depth_ratio > 0
        ):
            smooth_meta = metadata["temporal_smoothness"]
            smooth_valid = smooth_meta["valid"]
            smooth_valid = smooth_valid.unsqueeze(-1)

            flows_n = smooth_meta["flows"]
            masks_n = smooth_meta["masks"]
            smooth_reproj_losses, smooth_disparity_losses, smooth_depth_ratio_losses, scene_flow_neighbor = \
                self.smooth_scene_flow_loss(
                    points_cam,
                    pixels,
                    intrinsics_pair,
                    extrinsics,
                    flows_n,
                    masks_n,
                    smooth_valid,
                )

        B = points_cam_pair[0].shape[0]
        dtype = points_cam_pair[0].dtype

        batch_losses = {}
        scene_flow_loss_sum = 0.0

        # Static scene flow loss directly in 3D (not stable)
        if self.opt.lambda_scene_flow_static > 0:
            static_loss = (
                self.opt.lambda_scene_flow_static
                * torch.mean(torch.stack(static_losses, dim=-1), dim=-1)
                if len(static_losses) > 0
                else torch.zeros(B, dtype=dtype, device=_device)
            )
            batch_losses.update({"static": static_loss})
            scene_flow_loss_sum = scene_flow_loss_sum + static_loss

        # Smooth scene flow loss (on spatial reprojection errors)
        if self.opt.lambda_smooth_reprojection > 0:
            smooth_reproj_loss = (
                self.opt.lambda_smooth_reprojection
                * torch.mean(torch.stack(smooth_reproj_losses, dim=-1), dim=-1)
                if len(smooth_reproj_losses) > 0
                else torch.zeros(B, dtype=dtype, device=_device)
            )
            batch_losses.update({"smooth_reproj": smooth_reproj_loss})
            scene_flow_loss_sum = scene_flow_loss_sum + smooth_reproj_loss

        # Smooth scene flow loss (on disparity errors)
        if self.opt.lambda_smooth_disparity > 0:
            smooth_disparity_loss = (
                self.opt.lambda_smooth_disparity
                * torch.mean(torch.stack(smooth_disparity_losses, dim=-1), dim=-1)
                if len(smooth_disparity_losses) > 0
                else torch.zeros(B, dtype=dtype, device=_device)
            )
            batch_losses.update({"smooth_disparity": smooth_disparity_loss})
            scene_flow_loss_sum = scene_flow_loss_sum + smooth_disparity_loss

        # Smooth scene flow loss (on depth ratio)
        if self.opt.lambda_smooth_depth_ratio > 0:
            smooth_depth_ratio_loss = (
                torch.mean(torch.stack(smooth_depth_ratio_losses, dim=-1), dim=-1)
                if len(smooth_depth_ratio_losses) > 0
                else torch.zeros(B, dtype=dtype, device=_device)
            )
            batch_losses.update({"smooth_depth_ratio": smooth_depth_ratio_loss})
            scene_flow_loss_sum = scene_flow_loss_sum + smooth_depth_ratio_loss

        # List of scene flow maps for visualization
        scene_flow = scene_flow_pair + scene_flow_neighbor

        scene_flow_loss_mean = torch.mean(scene_flow_loss_sum)

        return scene_flow_loss_mean, batch_losses, scene_flow

    def __call__(
        self,
        depths,
        metadata,
    ):
        """Compute total loss.

        The network predicts a set of depths results. The number of samples, N, is
        not the batch_size, but computed based on the loss.
        The static scene flow loss requires pairs as samples, the N =2
        Adding temporal smoothness loss requires two triplet: (R-1, R, R+1) and (T-1, T, T+1).
        Thus, N = 6

        Args:
            depths (B, N, H, W):   predicted_depths
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include data as below. But each loss assumes more.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4),
                                  # (fx, fy, cx, cy) for each frame in pixels
                }

        Returns:
            loss: python scalar. And set self.total_loss
        """

        def squeeze(x):
            return x.reshape((-1,) + x.shape[2:])

        def unsqueeze(x, N):
            return x.reshape((-1, N) + x.shape[1:])

        depths = depths.unsqueeze(-3)
        intrinsics = metadata["intrinsics"]

        B, N, C, H, W = depths.shape

        # Pixel cooridnate
        pixels = pixel_grid(B * N, (H, W))

        if self.opt.recon != "colmap":
            # Warp map from spatial transformation
            warp_map = metadata["warp"].view(B * N, 2, H, W)  # (B*N, 2, H, W)
            warp_map[:, 0, :, :] = warp_map[:, 0, :, :] * (W / 2)
            warp_map[:, 1, :, :] = warp_map[:, 1, :, :] * (H / 2)

            # Apply spatial transformation
            pixels = pixels + warp_map

        points_cam = pixels_to_points(squeeze(intrinsics), squeeze(depths), pixels)
        pixels = unsqueeze(pixels, N)
        points_cam = unsqueeze(points_cam, N)

        return self.scene_flow_loss(points_cam, metadata, pixels)
