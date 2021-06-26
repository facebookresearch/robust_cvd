#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from collections import namedtuple
from typing import Optional
import torch
from utils import (
    geometry as geom,
)


Pose = namedtuple("Pose", ["R", "t"])


def make_pose(extrinsics: torch.Tensor) -> Pose:
    """
    extrinsics: (B, 3, 3)
    """
    return Pose(R=extrinsics[..., :-1], t=extrinsics[..., -1:])


def transpose(t: torch.Tensor):
    """
    t (B, M, N) -> (B, N, M)
    """
    return t.transpose(1, 2)


def relative_pose(pose_r: Pose, pose_t: Pose) -> Pose:
    """
    R_r * x_r + t_r = R_t * x_t + t_t
    Compute R, t s.t.,
        x_r = R * (x_t - t)
    """
    R = torch.bmm(transpose(pose_r.R), pose_t.R)
    t = torch.bmm(transpose(pose_t.R), pose_r.t - pose_t.t)
    return Pose(R=R, t=t)


def projection_matrix(intr: torch.Tensor) -> torch.Tensor:
    """
    Compute projection matrix K from intrinsics s.t.,
    [u, v, 1]' = K * [x, y, z]'
    Note that pixel coordinate v is flipped
    Args:
        intr (B, 4): [[fx, fy, cx, cy]]
    Returns:
        K (B, 3, 3)
    """
    assert intr.shape[1:] == (4,)
    fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]
    B = intr.shape[0]
    dtype = intr.dtype
    device = intr.device
    o = torch.ones((B,), dtype=dtype, device=device)
    z = torch.zeros((B,), dtype=dtype, device=device)
    K = torch.stack([
        -fx,  z, cx,
          z, fy, cy,
          z,  z,  o,
    ], dim=-1).reshape(B, 3, 3)
    return K


def cross_prod_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    Given vector t, compute matrix S corresponding to its cross product, i.e.,
    cross_product(t, x) = S * x.
    Args:
        t (B, 3, 1)
    Returns:
        S (B, 3, 3)
    """
    assert t.shape[1:] == (3, 1)
    tx, ty, tz = t[:, 0, :], t[:, 1, :], t[:, 2, :]
    B = t.shape[0]
    dtype = t.dtype
    device = t.device
    z = torch.zeros((B, 1), dtype=dtype, device=device)

    S = torch.cat([
        z, -tz, ty,
        tz, z, -tx,
        -ty, tx, z
    ], dim=-1).reshape(B, 3, 3)
    return S


def normalize_matrix(M: torch.Tensor):
    """
    Normalize matrix M s.t., M[-1, -1] == 1
    """
    assert M.shape[1:] == (3, 3)
    p = M[:, -1:, -1:]
    return M / p


def compute_essential_matrix(pose: Pose) -> torch.Tensor:
    """
    Given R, t s.t., x_r = R (x_t - t), see relative_pose,
    compute essential matrix E s.t.
        r_r^T E r_t = 0, where r_r and r_t are reference and target rays.
    """
    S = cross_prod_matrix(pose.t)
    E = torch.bmm(pose.R, S)
    E = normalize_matrix(E)
    return E


def compute_fundamental_matrix(
    pose: Pose, K_r: torch.Tensor, K_t: torch.Tensor
) -> torch.Tensor:
    """
    Given R, t s.t., x_r = R (x_t - t), see relative_pose,
    compute fundamental matrix E s.t.
        p_r^T F p_t = 0
    """
    E = compute_essential_matrix(pose)
    K_inv_r, K_inv_t = torch.inverse(K_r), torch.inverse(K_t)
    F = torch.bmm(torch.bmm(transpose(K_inv_r), E), K_inv_t)
    F = normalize_matrix(F)
    return F


def flatten(x: torch.Tensor) -> torch.Tensor:
    """
        Convert x of shape (B, C, ...) to (B, C, N)
    """
    return x.reshape(x.shape[:2] + (-1,))


def essential_constraint(
    E: torch.Tensor, ray_r: torch.Tensor, ray_t: torch.Tensor
) -> torch.Tensor:
    """
    Compute essential matrix s.t.,
        r_r^T E r_t =  0
    """
    def dot(x, y):
        # x.shape == y.shape = (B, 3, N)
        return torch.sum(x * y, dim=1)

    shape = ray_r.shape
    rrT_E = torch.bmm(transpose(flatten(ray_r)), E)  # B, N, 3
    rrT_E_rt = dot(transpose(rrT_E), flatten(ray_t))

    a_b = transpose(rrT_E[..., :2])
    line_norm = torch.sqrt(dot(a_b, a_b))
    dist = rrT_E_rt / line_norm
    return dist.reshape(shape[0], shape[-2], shape[-1])


def to_homo(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D vector x to 3D homogeneous coordiante.
    Args:
        x (B, 2, ...)
    Returns:
        (B, 3, ...)
    """
    assert x.shape[1] == 2
    o = torch.ones_like(x[:, -1:, ...])
    return torch.cat((x, o), dim=1)


def fundamental_constraint(
    F: torch.Tensor, pixel_r: torch.Tensor, pixel_t: torch.Tensor
) -> torch.Tensor:
    """
    Compute how much corresponding pixels pixel_r, pixel_r violates epipolar constraint.
    Args:
        F (B, 3, 3)
        pixel_r, pixel_t (B, 2, ...)
    Returns:
        (B, ...)
    """
    p_r = to_homo(pixel_r)
    p_t = to_homo(pixel_t)
    return essential_constraint(F, p_r, p_t)


def epipolar_lines(pixel_r: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
        Compute epipolar lines l = p_r^T * F
    """
    p_r = to_homo(pixel_r)
    shape = p_r.shape
    lines = torch.bmm(transpose(flatten(p_r)), F)  # B, N, 3
    return transpose(lines).reshape(shape)


def closest_point_on_line(
    line: torch.Tensor, p: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Find closest pixel on line to pixel p.
    Args:
        line: (B, 3, H, W) and the three channels are a,b,c corresponding to line
            ax+by+c=0
        p: (B, 2, H, W) pixels
    Returns:
        p': (B, 2, H, W) closest pixel position on the line
    let d = (b, -a), n = (a, b). then closest point
             1            [<d^T, p>]
    p' = --------- [d, n]
         (a^2+b^2)        [   -c   ]
    """
    assert line.shape[1] == 3 and p.shape[1] == 2
    line_f = transpose(flatten(line)).reshape(-1, 3, 1)  # (BxN ,3, 1)
    p_f = transpose(flatten(p)).reshape(-1, 2, 1)    # (BxN, 2, 1)
    a, b, c = line_f[:, :1], line_f[:, 1:2], line_f[:, 2:3]  # (BxN, 1, 1)
    sq_norm = a * a + b * b  # (BxN, 1, 1)
    sq_norm[sq_norm < eps] = float('nan')
    d = torch.cat([b, -a], dim=1)
    n = line_f[:, :2]
    A = torch.cat((d, n), dim=-1)  # [d, n] (BxN, 2, 2)
    dTp = torch.bmm(transpose(d), p_f)
    bb = torch.cat((dTp, -c), dim=1)
    p_result = torch.bmm(A, bb) / sq_norm
    B, C, H, W = p.shape
    return p_result.reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)


def pixel_to_ray(
    K: torch.Tensor, pixel: torch.Tensor
) -> torch.Tensor:
    K_inv = torch.inverse(K)
    p = flatten(to_homo(pixel))
    ray = torch.bmm(K_inv, p)
    ray = ray / ray[:, -1:, ...]

    B, _, H, W = pixel.shape
    return ray.reshape((B, 3, H, W))


def compute_depth(
    pixel_r: torch.Tensor,
    pixel_t: torch.Tensor,
    K_r: torch.Tensor,
    K_t: torch.Tensor,
    pose: Pose,
    eps: float = 1e-3
) -> torch.Tensor:
    """
    Given pixel_r, pixel_t that agree with epipolar constraint, compute depth z
    from the reference viewpoint as follows.
        z*r_r = R(x_t - t).
        lambda * p_t = K_t*x_t
        => lambda p_t = z * K_t*R^T*r_r + K_t*t
        => z * (p_t x (K_t*R^T*r_r)) = -p_t x (K_t*t)
    Note that z is negative.
    """
    ray_r = pixel_to_ray(K_r, pixel_r)
    r_r = flatten(ray_r)  # B, 3, N

    KRT = torch.bmm(K_t, transpose(pose.R))  # B, 3, 3
    KRTr = torch.bmm(KRT, r_r)
    p_t = flatten(to_homo(pixel_t))
    lhs = torch.cross(p_t, KRTr, dim=1)

    Kt = torch.bmm(K_t, pose.t)  # B, 3, 1
    rhs = - torch.cross(p_t, Kt.expand_as(p_t))

    lhs[torch.abs(lhs) < eps] = float('nan')
    z = (rhs / lhs).reshape(ray_r.shape)
    return torch.mean(z, dim=1)


def compute_closest_epipolar_depth(
    pixel_r: torch.Tensor,
    pixel_t: torch.Tensor,
    pose: Pose,
    K_r: torch.Tensor,
    K_t: torch.Tensor,
    F: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Given pixel_r, pixel_t that not necessarily agree with epipolar constraint,
    compute the best approximation depth z that follows epipolar constraint.
    Note that z is negative.
    """
    if F is None:
        F = compute_fundamental_matrix(pose, K_r, K_t)
    line = epipolar_lines(pixel_r, F)
    pixel_t_closest = closest_point_on_line(line, pixel_t)
    z = compute_depth(pixel_r, pixel_t_closest, K_r, K_t, pose)
    return z


def epipolar_error_and_depth(
    pixel_r: torch.Tensor, pixel_t: torch.Tensor,
    pose_r: Pose, pose_t: Pose,
    K_r: torch.Tensor, K_t: torch.Tensor,
) -> torch.Tensor:
    """
    Given pixel_r, pixel_t that not necessarily agree with epipolar constraint,
    compute E_epipolar measuring the distance of pixel_t to corresponding epipolar line
    and approximate depth Z_epipolar that follows epipolar constraint.
    """
    pose = relative_pose(pose_r, pose_t)
    F = compute_fundamental_matrix(pose, K_r, K_t)

    # compute epipolar constraint error
    E_epipolar = fundamental_constraint(F, pixel_r, pixel_t)

    # compute depth from epipolar contraint
    Z_epipolar = compute_closest_epipolar_depth(pixel_r, pixel_t, pose, K_r, K_t, F=F)
    return E_epipolar, Z_epipolar


def epipolar_error_and_depth_from_flow(
    flow: torch.Tensor,
    extr_r: torch.Tensor, extr_t: torch.Tensor,
    intr_r: torch.Tensor, intr_t: torch.Tensor,
) -> torch.Tensor:
    """
    Given optical flow that not necessarily agree with epipolar constraint,
    compute E_epipolar measuring the distance of pixel_t to corresponding epipolar line
    and approximate depth Z_epipolar that follows epipolar constraint.
    """
    pose_r, pose_t = make_pose(extr_r), make_pose(extr_t)
    K_r, K_t = projection_matrix(intr_r), projection_matrix(intr_t)

    pixel_r = geom.pixel_grid(flow.shape[0], flow.shape[-2:])
    pixel_t = pixel_r + flow

    E_epipolar, Z_epipolar = epipolar_error_and_depth(
        pixel_r, pixel_t, pose_r, pose_t, K_r, K_t
    )
    return E_epipolar, Z_epipolar
