#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn as nn
from loss import distance


class DisparitySmoothLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.robust_dist = distance.create(opt.distance_type, opt)

    def forward(self, images, depths):
        """Spatial smooth Loss.

        This loss panelizes the L1-norm of the disparity gradient.
        The loss has edge-preserving weights extracted from the gradients of color image.

        Loss = |grad_x(Disp)|*exp(-|norm(grad_x(Img))|) + |grad_y(Disp)|*exp(-|norm(grad_y(Img))|)

        Args:
            images (B, N, 3, H, W): color images
            depths (B, N, H, W): predicted depth
        """

        disp_spatial_smooth_losses = []

        disparity = 1.0 / depths

        grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])

        grad_img_x = images[:, :, :, :, :-1] - images[:, :, :, :, 1:]
        grad_img_y = images[:, :, :, :-1, :] - images[:, :, :, 1:, :]
        grad_mag_x = torch.mean(torch.abs(grad_img_x), 2, keepdim=False)
        grad_mag_y = torch.mean(torch.abs(grad_img_y), 2, keepdim=False)

        grad_disp_x *= torch.exp(-grad_mag_x / self.opt.sigma_color_grad)
        grad_disp_y *= torch.exp(-grad_mag_y / self.opt.sigma_color_grad)

        B = images.shape[0]
        grad_disp_x_mean = torch.mean(grad_disp_x.reshape(B, -1), dim=1)
        grad_disp_y_mean = torch.mean(grad_disp_y.reshape(B, -1), dim=1)

        disp_spatial_smooth_losses.append(
            grad_disp_x_mean + grad_disp_y_mean
        )

        disp_spatial_smooth_loss = torch.mean(torch.stack(disp_spatial_smooth_losses, dim=-1), dim=-1)
        disp_spatial_smooth_loss *= self.opt.lambda_disparity_smooth

        batch_losses = {"disparity_smooth": disp_spatial_smooth_loss, }

        return torch.mean(disp_spatial_smooth_loss), batch_losses
