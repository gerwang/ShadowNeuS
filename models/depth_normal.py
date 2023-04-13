import numpy as np
import torch


def get_depth_map_faces(verts_map, threshold):
    H, W = verts_map.shape[:2]
    idx_0 = np.meshgrid(np.arange(0, H - 1), np.arange(0, W - 1), indexing='ij')
    idx_1 = np.meshgrid(np.arange(1, H), np.arange(0, W - 1), indexing='ij')
    idx_2 = np.meshgrid(np.arange(0, H - 1), np.arange(1, W), indexing='ij')
    idx_3 = np.meshgrid(np.arange(1, H), np.arange(1, W), indexing='ij')
    verts_0 = verts_map[tuple(idx_0)]
    verts_1 = verts_map[tuple(idx_1)]
    verts_2 = verts_map[tuple(idx_2)]
    verts_3 = verts_map[tuple(idx_3)]
    edge_01 = verts_1 - verts_0
    edge_02 = verts_2 - verts_0
    edge_31 = verts_1 - verts_3
    edge_32 = verts_2 - verts_3
    area_0 = np.linalg.norm(np.cross(edge_01, edge_02, axis=-1), axis=-1)
    area_3 = np.linalg.norm(np.cross(edge_31, edge_32, axis=-1), axis=-1)
    valid_idx_0 = area_0 < threshold
    valid_idx_3 = area_3 < threshold
    faces_0 = idx_0[0] * W + idx_0[1]
    faces_1 = idx_1[0] * W + idx_1[1]
    faces_2 = idx_2[0] * W + idx_2[1]
    faces_3 = idx_3[0] * W + idx_3[1]
    faces = np.concatenate([
        np.stack([faces_0, faces_1, faces_2], axis=-1)[valid_idx_0],
        np.stack([faces_3, faces_2, faces_1], axis=-1)[valid_idx_3],
    ])
    return faces


def norm_diff(normal_hat, norm_gt, silhouette=None):
    """Tensor Dim: NxCxHxW"""
    if norm_gt.ndim != 4:
        norm_gt = norm_gt.unsqueeze(0)
    if normal_hat.ndim != 4:
        normal_hat = normal_hat.unsqueeze(0)
    if norm_gt.shape[1] != 3:
        print("Warning: norm_diff received wrong shape for norm_gt")
        norm_gt = norm_gt.permute(0, 3, 1, 2)
    if normal_hat.shape[1] != 3:
        print("Warning: norm_diff received wrong shape for normal_hat")
        normal_hat = normal_hat.permute(0, 3, 1, 2)
    if silhouette is None:
        silhouette = torch.ones((1, 1, norm_gt.shape[2], norm_gt.shape[3])).to(norm_gt.device)
    elif silhouette.ndim != 4:
        silhouette = silhouette.reshape(1, 1, normal_hat.shape[2], normal_hat.shape[3])

    dot_product = (norm_gt * normal_hat).sum(1).clamp(-1, 1)
    error_map = torch.acos(dot_product)  # [-pi, pi]
    angular_map = error_map * 180.0 / np.pi
    angular_map = angular_map * silhouette.narrow(1, 0, 1).squeeze(1)

    valid_mask = silhouette.narrow(1, 0, 1).squeeze(1).bool()
    valid = valid_mask.sum()
    ang_valid = angular_map[valid_mask]
    n_err_mean = ang_valid.sum() / valid
    return angular_map.squeeze(), n_err_mean
