#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch

def select_tensors(x):
    """
    x (B, N, C, H, W) -> (N, B, C, H, W)
    Each batch (B) is composed of a pair or more samples (N).
    """
    return x.transpose(0, 1)


def weighted_mse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    sq_error = torch.sum((input - target) ** 2, dim=dim, keepdim=True)  # BHW
    return torch.sum((weights_n * sq_error).reshape(B, -1), dim=1)


def weighted_rmse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar = weighted_mean(rmse_along_dim)
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    diff = torch.norm(input - target, dim=dim, keepdim=True)
    return torch.sum((weights_n * diff).reshape(B, -1), dim=1)


def weighted_mean_loss(x, weights, eps=1e-6):
    """
        Args:
            x (B, ...)
            weights (B, ...)

        Returns:
            a scalar
    """
    assert x.ndimension() == weights.ndimension() and x.shape[0] == weights.shape[0]
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    weighted_mean = torch.sum((weights_n * x).reshape(B, -1), dim=1)

    return weighted_mean
