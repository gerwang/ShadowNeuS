import cv2
import kornia
import math
import numpy as np
import torch
import torch.nn as nn
from icecream import ic

VERBOSE_MODE = False


def reparam_points(nondiff_points, nondiff_grads, nondiff_trgt_dirs, diff_sdf_vals):
    # note that flipping the direction of nondiff_trgt_dirs would not change this equations at all
    # hence we require dot >= 0
    dot = (nondiff_grads * nondiff_trgt_dirs).sum(dim=-1, keepdim=True)
    # assert (dot >= 0.).all(), 'dot>=0 not satisfied in reparam_points: {},{}'.format(dot.min().item(), dot.max().item())
    dot = torch.clamp(dot, min=1e-4)
    diff_points = nondiff_points - nondiff_trgt_dirs / dot * (diff_sdf_vals - diff_sdf_vals.detach())
    return diff_points


class RayTracer(nn.Module):
    def __init__(
            self,
            sdf_threshold=5.0e-5,
            sphere_tracing_iters=0,
            n_steps=256,
            n_outer_steps=0,
            max_num_pts=200000,
            mock_floor=True,
            test_mode=False,
    ):
        super().__init__()
        """sdf values of convergent points must be inside [-sdf_threshold, sdf_threshold]"""
        self.sdf_threshold = sdf_threshold
        # sphere tracing hyper-params
        self.sphere_tracing_iters = sphere_tracing_iters
        # dense sampling hyper-params
        self.n_steps = n_steps
        self.n_outer_steps = n_outer_steps

        self.max_num_pts = max_num_pts
        self.mock_floor = mock_floor
        self.test_mode = test_mode

    @torch.no_grad()
    def forward(self, sdf, ray_o, ray_d, min_dis, max_dis, min_dis_outer, max_dis_outer, work_mask, floor_dist):
        (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_start_sdf,
            acc_start_dis,
        ) = self.sphere_tracing(sdf, ray_o, ray_d, min_dis_outer, max_dis_outer, work_mask)
        sphere_tracing_cnt = convergent_mask.sum()

        sampler_work_mask = unfinished_mask_start
        sampler_cnt = 0
        if sampler_work_mask.sum() > 0:
            sampler_min_dis = min_dis[sampler_work_mask]
            sampler_min_dis_outer = min_dis_outer[sampler_work_mask]
            sampler_max_dis = max_dis[sampler_work_mask]
            sampler_max_dis_outer = max_dis_outer[sampler_work_mask]

            (sampler_convergent_mask, sampler_points, sampler_sdf, sampler_dis,) = self.ray_sampler(
                sdf,
                ray_o[sampler_work_mask],
                ray_d[sampler_work_mask],
                sampler_min_dis,
                sampler_max_dis,
                sampler_min_dis_outer,
                sampler_max_dis_outer,
            )

            convergent_mask[sampler_work_mask] = sampler_convergent_mask
            curr_start_points[sampler_work_mask] = sampler_points
            curr_start_sdf[sampler_work_mask] = sampler_sdf
            acc_start_dis[sampler_work_mask] = sampler_dis
            sampler_cnt = sampler_convergent_mask.sum()

        real_mask = convergent_mask.clone()
        real_points = curr_start_points.clone()
        if self.mock_floor and floor_dist is not None:  # mock non-convergent rays
            floor_points = ray_o + ray_d * floor_dist
            background_mask = ~convergent_mask
            convergent_mask[background_mask] = True
            curr_start_points[background_mask] = floor_points[background_mask]
            curr_start_sdf[background_mask] = 0
            acc_start_dis[background_mask] = floor_dist[background_mask, 0]

        if self.test_mode and floor_dist is not None:  # Remove floaters under the floor when visualizing novel views
            floor_points = ray_o + ray_d * floor_dist
            convergent_mask &= real_points[..., 2] > floor_points[..., 2] - 0.03

        ret_dict = {
            "convergent_mask": convergent_mask,
            "real_mask": real_mask,
            "points": curr_start_points,
            "real_points": real_points,
            "sdf": curr_start_sdf,
            "distance": acc_start_dis,
        }

        if VERBOSE_MODE:  # debug
            sdf_check = sdf(curr_start_points)
            ic(
                convergent_mask.sum() / convergent_mask.numel(),
                sdf_check[convergent_mask].min().item(),
                sdf_check[convergent_mask].max().item(),
            )
            debug_info = "Total,raytraced,convergent(sphere tracing+dense sampling): {},{},{} ({}+{})".format(
                work_mask.numel(),
                work_mask.sum(),
                convergent_mask.sum(),
                sphere_tracing_cnt,
                sampler_cnt,
            )
            ic(debug_info)
        return ret_dict

    def sphere_tracing(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask):
        """Run sphere tracing algorithm for max iterations"""
        iters = 0
        unfinished_mask_start = work_mask.clone()
        acc_start_dis = min_dis.clone()
        curr_start_points = ray_o + ray_d * acc_start_dis.unsqueeze(-1)
        curr_sdf_start = sdf(curr_start_points)
        while True:
            # Check convergence
            unfinished_mask_start = (
                    unfinished_mask_start & (curr_sdf_start.abs() > self.sdf_threshold) & (acc_start_dis < max_dis)
            )

            if iters == self.sphere_tracing_iters or unfinished_mask_start.sum() == 0:
                break
            iters += 1

            # Make step
            tmp = curr_sdf_start[unfinished_mask_start]
            acc_start_dis[unfinished_mask_start] += tmp
            curr_start_points[unfinished_mask_start] += ray_d[unfinished_mask_start] * tmp.unsqueeze(-1)
            curr_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        convergent_mask = (
                work_mask
                & ~unfinished_mask_start
                & (curr_sdf_start.abs() <= self.sdf_threshold)
                & (acc_start_dis < max_dis)
        )
        return (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_sdf_start,
            acc_start_dis,
        )

    def ray_sampler(self, sdf, ray_o, ray_d, min_dis, max_dis, min_dis_outer, max_dis_outer):
        """Sample the ray in a given range and perform rootfinding on ray segments which have sign transition"""
        n_inner_steps = self.n_steps - self.n_outer_steps
        intervals_dis = (
            torch.linspace(0, 1, steps=n_inner_steps).float().to(min_dis.device).view(1, n_inner_steps)
        )  # [1, n_steps]
        intervals_dis = min_dis.unsqueeze(-1) + intervals_dis * (
                max_dis.unsqueeze(-1) - min_dis.unsqueeze(-1)
        )  # [n_valid, n_steps]
        if self.n_outer_steps > 0:
            half_outer_steps = self.n_outer_steps // 2
            intervals_dis_outer_near = (
                torch.linspace(0, 1, steps=half_outer_steps).float().to(min_dis.device).view(1, half_outer_steps)
            )  # [1, half_outer_steps]
            intervals_dis_outer_near = min_dis_outer.unsqueeze(-1) + intervals_dis_outer_near * (
                    min_dis.unsqueeze(-1) - min_dis_outer.unsqueeze(-1)
            )  # [n_valid, half_outer_steps]
            intervals_dis_outer_far = (
                torch.linspace(0, 1, steps=half_outer_steps).float().to(min_dis.device).view(1, half_outer_steps)
            )  # [1, half_outer_steps]
            intervals_dis_outer_far = max_dis.unsqueeze(-1) + intervals_dis_outer_far * (
                    max_dis_outer.unsqueeze(-1) - max_dis.unsqueeze(-1)
            )  # [n_valid, half_outer_steps]
            intervals_dis = torch.cat([intervals_dis_outer_near, intervals_dis, intervals_dis_outer_far], dim=-1)
        points = ray_o.unsqueeze(-2) + ray_d.unsqueeze(-2) * intervals_dis.unsqueeze(-1)  # [n_valid, n_steps, 3]

        sdf_val = []
        for pnts in torch.split(points.reshape(-1, 3), self.max_num_pts, dim=0):
            sdf_val.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val, dim=0).reshape(-1, self.n_steps)

        _, min_idx = sdf_val.abs().min(dim=-1)
        # To be returned
        sampler_pts = points.gather(dim=1, index=min_idx[:, None, None].repeat(1, 1, points.shape[-1]))[:, 0].clone()
        sampler_sdf = sdf_val.gather(dim=1, index=min_idx[:, None])[:, 0].clone()
        sampler_dis = intervals_dis.gather(dim=1, index=min_idx[:, None])[:, 0].clone()

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).float().to(sdf_val.device).reshape(
            1, self.n_steps
        )
        # return first negative sdf point if exists
        min_val, min_idx = torch.min(tmp, dim=-1)
        rootfind_work_mask = (min_val < 0.0) & (min_idx >= 1)
        n_rootfind = rootfind_work_mask.sum()
        if n_rootfind > 0:
            # [n_rootfind, 1]
            min_idx = min_idx[rootfind_work_mask].unsqueeze(-1)
            z_low = torch.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(
                -1
            )  # [n_rootfind, ]
            # [n_rootfind, ]; > 0
            sdf_low = torch.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(-1)
            z_high = torch.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx).squeeze(
                -1
            )  # [n_rootfind, ]
            # [n_rootfind, ]; < 0
            sdf_high = torch.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx).squeeze(-1)

            p_pred, z_pred, sdf_pred = self.rootfind(
                sdf,
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                ray_o[rootfind_work_mask],
                ray_d[rootfind_work_mask],
            )

            sampler_pts[rootfind_work_mask] = p_pred
            sampler_sdf[rootfind_work_mask] = sdf_pred
            sampler_dis[rootfind_work_mask] = z_pred

        return rootfind_work_mask, sampler_pts, sampler_sdf, sampler_dis

    def rootfind(self, sdf, f_low, f_high, d_low, d_high, ray_o, ray_d):
        """binary search the root"""
        work_mask = (f_low > 0) & (f_high < 0)
        d_mid = (d_low + d_high) / 2.0
        i = 0
        while work_mask.any():
            p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
            f_mid = sdf(p_mid)
            ind_low = f_mid > 0
            ind_high = f_mid <= 0
            if ind_low.sum() > 0:
                d_low[ind_low] = d_mid[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if ind_high.sum() > 0:
                d_high[ind_high] = d_mid[ind_high]
                f_high[ind_high] = f_mid[ind_high]
            d_mid = (d_low + d_high) / 2.0
            work_mask &= (d_high - d_low) > 2 * self.sdf_threshold
            i += 1
        p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
        f_mid = sdf(p_mid)
        return p_mid, d_mid, f_mid


@torch.no_grad()
def intersect_sphere(ray_o, ray_d, r, keepdim=False):
    """
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    """
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d

    tmp = r * r - torch.sum(p * p, dim=-1)
    mask_intersect = tmp > 0.0
    d2 = torch.sqrt(torch.clamp(tmp, min=0.0)) / torch.norm(ray_d, dim=-1)

    near = d1 - d2
    far = d1 + d2
    if keepdim:
        near = near.unsqueeze(-1)
        far = far.unsqueeze(-1)
    return mask_intersect, near, far


class Camera(object):
    def __init__(self, W, H, K=None, W2C=None, K_inv=None, C2W=None):
        """
        W, H: int
        K, W2C: 4x4 tensor
        """
        self.W = W
        self.H = H
        if K is None:
            K = torch.inverse(K_inv)
        if K_inv is None:
            K_inv = torch.inverse(K)
        if W2C is None:
            W2C = torch.inverse(C2W)
        if C2W is None:
            C2W = torch.inverse(W2C)
        self.K = K
        self.W2C = W2C
        self.K_inv = K_inv
        self.C2W = C2W
        self.device = self.K.device

    def get_rays(self, uv):
        """
        uv: [..., 2]
        """
        dots_sh = list(uv.shape[:-1])

        uv = uv.view(-1, 2)
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)
        ray_d = torch.matmul(
            torch.matmul(uv, self.K_inv[:3, :3].transpose(1, 0)),
            self.C2W[:3, :3].transpose(1, 0),
        ).reshape(
            dots_sh
            + [
                3,
            ]
        )

        ray_d_norm = ray_d.norm(dim=-1)
        ray_d = ray_d / ray_d_norm.unsqueeze(-1)

        ray_o = (
            self.C2W[:3, 3]
            .unsqueeze(0)
            .expand(uv.shape[0], -1)
            .reshape(
                dots_sh
                + [
                    3,
                ]
            )
        )
        return ray_o, ray_d, ray_d_norm

    def get_camera_origin(self, prefix_shape=None):
        ray_o = self.C2W[:3, 3]
        if prefix_shape is not None:
            prefix_shape = list(prefix_shape)
            ray_o = ray_o.view([1, ] * len(prefix_shape) + [3, ]).expand(
                prefix_shape
                + [
                    3,
                ]
            )
        return ray_o

    def get_pixels(self):
        tx = torch.arange(self.W)
        ty = torch.arange(self.H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='xy')
        pixels = torch.stack([pixels_x, pixels_y], dim=-1)
        return pixels

    def get_uv(self):
        return self.get_pixels().float() + 0.5

    def project(self, points):
        """
        points: [..., 3]
        """
        dots_sh = list(points.shape[:-1])

        points = points.view(-1, 3)
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        uv = torch.matmul(
            torch.matmul(points, self.W2C.transpose(1, 0)),
            self.K.transpose(1, 0),
        )
        uv = uv[:, :2] / uv[:, 2:3]

        uv = uv.view(
            dots_sh
            + [
                2,
            ]
        )
        return uv

    def crop_region(self, trgt_W, trgt_H, center_crop=False, ul_corner=None, image=None):
        K = self.K.clone()
        if ul_corner is not None:
            ul_col, ul_row = ul_corner
        elif center_crop:
            ul_col = self.W // 2 - trgt_W // 2
            ul_row = self.H // 2 - trgt_H // 2
        else:
            ul_col = np.random.randint(0, self.W - trgt_W)
            ul_row = np.random.randint(0, self.H - trgt_H)
        # modify K
        K[0, 2] -= ul_col
        K[1, 2] -= ul_row

        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        if image is not None:
            assert image.shape[0] == self.H and image.shape[1] == self.W, "image size does not match specfied size"
            image = image[ul_row: ul_row + trgt_H, ul_col: ul_col + trgt_W]
        return camera, image, (ul_col, ul_row)

    def resize(self, factor, image=None):
        trgt_H, trgt_W = int(self.H * factor), int(self.W * factor)
        K = self.K.clone()
        K[0, :3] *= trgt_W / self.W
        K[1, :3] *= trgt_H / self.H
        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        if image is not None:
            device = image.device
            image = cv2.resize(image.detach().cpu().numpy(), (trgt_W, trgt_H), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image).to(device)
        return camera, image


@torch.no_grad()
def raytrace_pixels(sdf_network, raytracer, uv, camera, bound_r, outer_bound_r, floor_solver, mask=None,
                    max_num_rays=200000, cam_mask=None, mock_inner=False):
    if mask is None:
        mask = torch.ones_like(uv[..., 0]).bool()
    if cam_mask is None:
        cam_mask = mask

    dots_sh = list(uv.shape[:-1])

    ray_o, ray_d, ray_d_norm = camera.get_rays(uv)
    sdf = lambda x: sdf_network(x)[..., 0]

    merge_results = None
    for ray_o_split, ray_d_split, ray_d_norm_split, mask_split, cam_mask_split in zip(
            torch.split(ray_o.view(-1, 3), max_num_rays, dim=0),
            torch.split(ray_d.view(-1, 3), max_num_rays, dim=0),
            torch.split(ray_d_norm.view(-1, ), max_num_rays, dim=0),
            torch.split(mask.view(-1, ), max_num_rays, dim=0),
            torch.split(cam_mask.view(-1, ), max_num_rays, dim=0),
    ):
        mask_intersect_split, min_dis_split, max_dis_split = intersect_sphere(ray_o_split, ray_d_split, r=bound_r)
        mask_intersect_outer, min_dis_outer, max_dis_outer = intersect_sphere(ray_o_split, ray_d_split, r=outer_bound_r)
        min_dis_split.clamp_(min=0.0)
        min_dis_outer.clamp_(min=0.0)
        no_inner_mask = ~mask_intersect_split
        half_outer_ratio = 0.5 * raytracer.n_outer_steps / raytracer.n_steps
        dis_outer = max_dis_outer[no_inner_mask] - min_dis_outer[no_inner_mask]
        min_dis_split[no_inner_mask] = min_dis_outer[no_inner_mask] + dis_outer * half_outer_ratio
        max_dis_split[no_inner_mask] = min_dis_outer[no_inner_mask] + dis_outer * (1.0 - half_outer_ratio)

        if floor_solver is not None:
            floor_dist_split = floor_solver.get_floor_distance(ray_o_split, ray_d_split)
            if floor_solver.clip_flag:
                relax_floor_dist_split = floor_dist_split[..., 0] + 0.1
                min_dis_split.clamp_(max=relax_floor_dist_split)
                max_dis_split.clamp_(max=relax_floor_dist_split)
        else:
            floor_dist_split = None

        work_mask = mask_intersect_outer & mask_split
        if mock_inner:
            work_mask = work_mask & cam_mask_split
        results = raytracer(
            sdf,
            ray_o_split,
            ray_d_split,
            min_dis_split,
            max_dis_split,
            min_dis_outer,
            max_dis_outer,
            work_mask,
            floor_dist_split,
        )
        results["depth"] = results["distance"] / ray_d_norm_split

        if merge_results is None:
            merge_results = dict(
                [
                    (
                        x,
                        [
                            results[x],
                        ],
                    )
                    for x in results.keys()
                    if isinstance(results[x], torch.Tensor)
                ]
            )
        else:
            for x in results.keys():
                merge_results[x].append(results[x])  # gpu

    for x in list(merge_results.keys()):
        results = torch.cat(merge_results[x], dim=0).reshape(
            dots_sh
            + [
                -1,
            ]
        )
        if results.shape[-1] == 1:
            results = results[..., 0]
        merge_results[x] = results  # gpu

    # append more results
    merge_results.update(
        {
            "uv": uv,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "ray_d_norm": ray_d_norm,
        }
    )
    return merge_results


def unique(x, dim=-1):
    """
    return: unique elements in x, and their original indices in x
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


@torch.no_grad()
def locate_edge_points(
        camera, pixels, walk_start_points, sdf_network, max_step, step_size, dot_threshold,
        max_num_rays=200000, mask=None, full_camera=False, valid_dis=5e-4, momentum=0.9,
):
    """
    walk on the surface to locate 3d edge points with high precision
    full_camera indicates whether the `pixels` are all the pixels on the camera. If True, accelerate indexing edge_pixel_idx
    """
    if mask is None:
        mask = torch.ones_like(walk_start_points[..., 0]).bool()

    walk_finish_points = walk_start_points.clone()
    walk_edge_found_mask = mask.clone()
    n_valid = mask.sum()
    if n_valid > 0:
        dots_sh = list(walk_start_points.shape[:-1])

        walk_finish_points_valid = []
        walk_edge_found_mask_valid = []
        for cur_points_split in torch.split(walk_start_points[mask].clone().view(-1, 3).detach(), max_num_rays, dim=0):
            walk_edge_found_mask_split = torch.zeros_like(cur_points_split[..., 0]).bool()
            not_found_mask_split = ~walk_edge_found_mask_split

            ray_o_split = camera.get_camera_origin(prefix_shape=cur_points_split.shape[:-1])

            max_pixel_distance = 1
            orig_pixels_split = camera.project(cur_points_split).floor().long()

            i = 0
            history_walkdir = torch.zeros_like(cur_points_split)
            while True:
                cur_viewdir_split = ray_o_split[not_found_mask_split] - cur_points_split[not_found_mask_split]
                cur_viewdir_split = cur_viewdir_split / (cur_viewdir_split.norm(dim=-1, keepdim=True) + 1e-10)
                cur_sdf_split, _, cur_normal_split = sdf_network.get_all(
                    cur_points_split[not_found_mask_split].view(-1, 3),
                    is_training=False,
                )
                cur_normal_split = cur_normal_split / (cur_normal_split.norm(dim=-1, keepdim=True) + 1e-10)
                # regularize walk direction such that we don't get far away from the zero iso-surface
                cur_points_split[not_found_mask_split] -= cur_sdf_split * cur_normal_split
                cur_pixels_split = camera.project(cur_points_split).floor().long()
                pixel_distance_split = (orig_pixels_split - cur_pixels_split).abs().max(dim=-1).values

                dot_split = (cur_normal_split * cur_viewdir_split).sum(dim=-1)
                tmp_not_converge_mask = (dot_split.abs() > dot_threshold) & (
                        pixel_distance_split[not_found_mask_split] <= max_pixel_distance)
                walk_edge_found_mask_split[not_found_mask_split] = ~tmp_not_converge_mask
                not_found_mask_split = ~walk_edge_found_mask_split

                if i >= max_step or not_found_mask_split.sum() == 0:
                    break

                cur_walkdir_split = cur_normal_split - cur_viewdir_split / dot_split.unsqueeze(-1)
                cur_walkdir_split = cur_walkdir_split / (cur_walkdir_split.norm(dim=-1, keepdim=True) + 1e-10)
                walkdir_valid = (cur_sdf_split.abs() < valid_dis).float()
                cur_walkdir_split = cur_walkdir_split * walkdir_valid
                history_walkdir[not_found_mask_split] = \
                    history_walkdir[not_found_mask_split] * momentum + \
                    (step_size * cur_walkdir_split)[tmp_not_converge_mask] * (1.0 - momentum)
                cur_points_split[not_found_mask_split] += history_walkdir[not_found_mask_split]

                i += 1

            walk_finish_points_valid.append(cur_points_split)
            walk_edge_found_mask_split &= pixel_distance_split <= max_pixel_distance
            walk_edge_found_mask_valid.append(walk_edge_found_mask_split)

        walk_finish_points[mask] = torch.cat(walk_finish_points_valid, dim=0)
        walk_edge_found_mask[mask] = torch.cat(walk_edge_found_mask_valid, dim=0)
        walk_finish_points = walk_finish_points.reshape(
            dots_sh
            + [
                3,
            ]
        )
        walk_edge_found_mask = walk_edge_found_mask.reshape(dots_sh)

    edge_points = walk_finish_points[walk_edge_found_mask]
    edge_mask = torch.zeros_like(pixels[..., 0], dtype=torch.bool)
    edge_uv = torch.zeros_like(edge_points[..., :2])
    update_pixels = torch.Tensor([]).long().to(walk_finish_points.device)
    if walk_edge_found_mask.any():
        # filter out edge points out of camera's fov;
        # if there are multiple edge points mapping to the same pixel, only keep one
        edge_uv = camera.project(edge_points)
        update_pixels = torch.floor(edge_uv.detach()).long()
        if full_camera:
            mask = (update_pixels[..., 0] >= 0) & (update_pixels[..., 0] < camera.W) & \
                   (update_pixels[..., 1] >= 0) & (update_pixels[..., 1] < camera.H)
            update_pixels = update_pixels[:, 1] * camera.W + update_pixels[:, 0]
            mask = mask.reshape(update_pixels.shape)
        else:
            update_pixels_1d = update_pixels[:, 1] * camera.W + update_pixels[:, 0]
            pixels_1d = pixels.view(-1, 2)
            pixels_1d = pixels_1d[:, 1] * camera.W + pixels_1d[:, 0]
            pixel_match_mask = pixels_1d[None, :] == update_pixels_1d[:, None]  # at most 1024 x 1024
            mask, update_pixels = pixel_match_mask.max(dim=-1)  # assume unique `pixels`
        update_pixels, edge_points, edge_uv = update_pixels[mask], edge_points[mask], edge_uv[mask]
        if mask.any():
            update_pixels, unique_idx = unique(update_pixels, dim=0)
            # assert update_pixels.shape == unique_idx.shape, f"{update_pixels.shape},{unique_idx.shape}"
            edge_points = edge_points[unique_idx]
            edge_uv = edge_uv[unique_idx]

            edge_mask.view(-1)[update_pixels] = True
        # edge_cnt = edge_mask.sum()
        # assert (
        #     edge_cnt == edge_points.shape[0]
        # ), f"{edge_cnt},{edge_points.shape},{edge_uv.shape},{update_pixels.shape},{torch.unique(update_pixels).shape},{update_pixels.min()},{update_pixels.max()}"
        # assert (
        #     edge_cnt == edge_uv.shape[0]
        # ), f"{edge_cnt},{edge_points.shape},{edge_uv.shape},{update_pixels.shape},{torch.unique(update_pixels).shape}"

    # ic(edge_mask.shape, edge_points.shape, edge_uv.shape)
    results = {"edge_mask": edge_mask, "edge_points": edge_points, "edge_uv": edge_uv, "edge_pixel_idx": update_pixels}

    if VERBOSE_MODE:  # debug
        edge_angles = torch.zeros_like(edge_mask).float()
        edge_sdf = torch.zeros_like(edge_mask).float().unsqueeze(-1)
        if edge_mask.any():
            ray_o = camera.get_camera_origin(prefix_shape=edge_points.shape[:-1])
            edge_viewdir = ray_o - edge_points
            edge_viewdir = edge_viewdir / (edge_viewdir.norm(dim=-1, keepdim=True) + 1e-10)
            with torch.enable_grad():
                edge_sdf_vals, _, edge_normals = sdf_network.get_all(edge_points, is_training=False)
            edge_normals = edge_normals / (edge_normals.norm(dim=-1, keepdim=True) + 1e-10)
            edge_dot = (edge_viewdir * edge_normals).sum(dim=-1)
            # edge_angles[edge_mask] = torch.rad2deg(torch.acos(edge_dot))
            # edge_sdf[edge_mask] = edge_sdf_vals
            edge_angles.view(-1)[update_pixels] = torch.rad2deg(torch.acos(edge_dot))
            edge_sdf.view(-1)[update_pixels] = edge_sdf_vals.squeeze(-1)

        results.update(
            {
                "walk_edge_found_mask": walk_edge_found_mask,
                "edge_angles": edge_angles,
                "edge_sdf": edge_sdf,
            }
        )

    return results


def pad_pixels(uv, camera):  # (n, 2)
    uv_left = uv - torch.tensor([1, 0])
    uv_right = uv + torch.tensor([1, 0])
    uv_row = torch.stack([uv_left, uv, uv_right], dim=0)
    uv_up = uv_row - torch.tensor([0, 1])
    uv_down = uv_row + torch.tensor([0, 1])
    uv_all = torch.cat([uv_left[None, ...], uv_right[None, ...], uv_up, uv_down], dim=0)
    # uv_all[..., 0].clamp_(0, camera.W - 1)
    # uv_all[..., 1].clamp_(0, camera.H - 1) do not clamp pixels, allow ray outside camera fov
    return uv_all


def combine_x(x, x_padded):
    """
    x: [...]
    x_padded: [8, ...]
    return: [3, 3, ...]
    """
    x_left, x_right, x_up, x_down = x_padded.split([1, 1, 3, 3], dim=0)
    x_row = torch.cat([x_left, x[None, ...], x_right], dim=0)
    x_all = torch.stack([x_up, x_row, x_down], dim=0)
    return x_all


@torch.no_grad()
def raytrace_camera(
        camera,
        pixels,
        sdf_network,
        raytracer,
        bound_r,
        outer_bound_r,
        floor_solver,
        max_num_rays=200000,
        detect_edges=False,
        full_camera=False,
        use_padded=False,
        cam_mask=None,
        mock_inner=False,
):
    sobel_threshold = 1e-2
    results = raytrace_pixels(sdf_network, raytracer, pixels.float() + 0.5, camera, bound_r=bound_r,
                              outer_bound_r=outer_bound_r, floor_solver=floor_solver, max_num_rays=max_num_rays,
                              cam_mask=cam_mask, mock_inner=mock_inner)
    results["depth"] *= results["real_mask"].float()

    if detect_edges:
        if full_camera:
            depth = results["depth"]
            convergent_mask = results["real_mask"]
            depth_grad_norm = kornia.filters.sobel(depth[None, None, ...])[0, 0]
            depth_edge_mask = (depth_grad_norm > sobel_threshold) & convergent_mask
            walk_start_points = results['points']
        elif use_padded:
            pixels_padded = pad_pixels(pixels, camera)
            if cam_mask is not None:
                cam_mask_padded = cam_mask.unsqueeze(0).expand(8, *cam_mask.shape).contiguous()
            else:
                cam_mask_padded = None
            results_padded = raytrace_pixels(sdf_network, raytracer, pixels_padded.float() + 0.5, camera,
                                             bound_r=bound_r, outer_bound_r=outer_bound_r,
                                             floor_solver=floor_solver,
                                             max_num_rays=max_num_rays,
                                             cam_mask=cam_mask_padded,
                                             mock_inner=mock_inner)
            results_padded["depth"] *= results_padded["real_mask"].float()
            depth_combined = combine_x(results['depth'], results_padded['depth'])
            convergent_mask_combined = combine_x(results['real_mask'], results_padded['real_mask'])
            depth_grad_norm = kornia.filters.sobel(depth_combined.view(
                *depth_combined.shape[:2], -1).permute(2, 0, 1)[None, ...])[0, :, 1, 1].reshape(results['depth'].shape)
            depth_grad_norm_padded = depth_grad_norm[None, None, ...].expand(3, 3, *depth_grad_norm.shape)
            depth_edge_mask = (depth_grad_norm_padded > sobel_threshold) & convergent_mask_combined
            walk_start_points = combine_x(results['points'], results_padded['points'])
        else:
            depth_edge_mask = results['real_mask']  # detect edge on all convergent points
            walk_start_points = results['points']

        sqrt_resolution_level = int(math.ceil(math.sqrt(math.sqrt(800 * 800 / (camera.W * camera.H)))))

        results.update(
            locate_edge_points(
                camera,
                pixels,
                walk_start_points,
                sdf_network,
                max_step=16,
                step_size=1e-3 * bound_r * sqrt_resolution_level,
                dot_threshold=5e-2 * sqrt_resolution_level,
                max_num_rays=max_num_rays,
                mask=depth_edge_mask,
                full_camera=full_camera,
            )
        )
        results["convergent_mask"] &= ~results["edge_mask"]

        # if VERBOSE_MODE:  # debug
        #     results.update({"depth_grad_norm": depth_grad_norm, "depth_edge_mask": depth_edge_mask})

    return results


def render_normal_and_color(
        results,
        sdf_network,
        color_network_dict,
        renderer,
        use_white_bkgd,
        do_reparam,
        is_training=False,
        max_num_pts=320000,
):
    """
    results: returned by raytrace_pixels function

    render interior and freespace pixels
    note: predicted color is black for freespace pixels
    """
    dots_sh = list(results["convergent_mask"].shape)

    merge_render_results = None
    diff_points = None
    for points_split, ray_d_split, ray_o_split, mask_split, real_mask_split in zip(
            torch.split(results["points"].view(-1, 3), max_num_pts, dim=0),
            torch.split(results["ray_d"].view(-1, 3), max_num_pts, dim=0),
            torch.split(results["ray_o"].view(-1, 3), max_num_pts, dim=0),
            torch.split(results["convergent_mask"].view(-1), max_num_pts, dim=0),
            torch.split(results["real_mask"].view(-1), max_num_pts, dim=0),
    ):
        diff_points_split = points_split.clone() if is_training else None
        if mask_split.any():
            points_split, ray_d_split, ray_o_split, real_mask_split = (
                points_split[mask_split],
                ray_d_split[mask_split],
                ray_o_split[mask_split],
                real_mask_split[mask_split],
            )
            sdf_split, feature_split, normal_split = sdf_network.get_all(points_split, is_training=is_training)
            if is_training and real_mask_split.any():
                nondiff_points = points_split.clone()
                points_split = points_split.clone()
                points_split[real_mask_split] = reparam_points(points_split[real_mask_split],
                                                               normal_split[real_mask_split].detach(),
                                                               -ray_d_split[real_mask_split].detach(),
                                                               sdf_split[real_mask_split])
                diff_points_split[mask_split] = points_split
                if not do_reparam:
                    points_split = nondiff_points
                # reparam the normal and feature to make it differentiable
                sdf_split[real_mask_split], feature_split[real_mask_split], normal_split[
                    real_mask_split] = sdf_network.get_all(points_split[real_mask_split], is_training=is_training)
        else:
            points_split, ray_d_split, ray_o_split, normal_split, feature_split = (
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
            )

        with torch.set_grad_enabled(is_training):
            render_results = renderer.render_fn(
                mask_split,
                sdf_network,
                color_network_dict,
                ray_o_split,
                ray_d_split,
                points_split,
                normal_split,
                feature_split,
                is_training,
                use_white_bkgd,
            )

            if merge_render_results is None:
                merge_render_results = dict(
                    [
                        (
                            x,
                            [
                                render_results[x],
                            ],
                        )
                        for x in render_results.keys()
                    ]
                )
            else:
                for x in render_results.keys():
                    merge_render_results[x].append(render_results[x])
            if is_training:
                if diff_points is None:
                    diff_points = [diff_points_split]
                else:
                    diff_points.append(diff_points_split)

    for x in list(merge_render_results.keys()):
        tmp = torch.cat(merge_render_results[x], dim=0)
        tmp = tmp.reshape(*dots_sh, *tmp.shape[1:])
        merge_render_results[x] = tmp
    if is_training:
        diff_points = torch.cat(diff_points, dim=0)
        diff_points = diff_points.reshape(*dots_sh, *diff_points.shape[1:])
        merge_render_results['diff_points'] = diff_points

    results.update(merge_render_results)


def render_edge_pixels(
        results,
        camera,
        sdf_network,
        raytracer,
        color_network_dict,
        renderer,
        bound_r,
        outer_bound_r,
        floor_solver,
        use_white_bkgd,
        max_num_rays,
        do_reparam,
        is_training=False,
        light_batch_size=1,
        cam_mask=None,
        mock_inner=False,
):
    edge_mask, edge_points, edge_uv, edge_pixel_idx = (
        results["edge_mask"],
        results["edge_points"],
        results["edge_uv"],
        results["edge_pixel_idx"],
    )
    edge_pixel_center = torch.floor(edge_uv) + 0.5

    edge_sdf, _, edge_grads = sdf_network.get_all(edge_points, is_training=is_training)
    edge_normals = edge_grads.detach() / (edge_grads.detach().norm(dim=-1, keepdim=True) + 1e-10)
    if is_training:
        edge_points = reparam_points(edge_points, edge_grads.detach(), edge_normals, edge_sdf)
        edge_uv = camera.project(edge_points)

    edge_normals2d = torch.matmul(edge_normals, camera.W2C[:3, :3].transpose(1, 0))[:, :2]
    edge_normals2d = edge_normals2d / (edge_normals2d.norm(dim=-1, keepdim=True) + 1e-10)

    # sample a point on both sides of the edge
    # approximately think of each pixel as being approximately a circle with radius 0.707=sqrt(2)/2
    pixel_radius = 0.707
    pos_side_uv = edge_pixel_center - pixel_radius * edge_normals2d
    neg_side_uv = edge_pixel_center + pixel_radius * edge_normals2d

    dot2d = torch.sum((edge_uv - edge_pixel_center) * edge_normals2d, dim=-1)
    alpha = 2 * torch.arccos(torch.clamp(dot2d / pixel_radius, min=0.0, max=1.0))
    pos_side_weight = 1.0 - (alpha - torch.sin(alpha)) / (2.0 * np.pi)

    # render positive-side and negative-side colors by raytracing; speed up using edge mask
    if cam_mask is not None:
        edge_cam_mask = cam_mask.view(-1)[edge_pixel_idx]
    else:
        edge_cam_mask = None
    pos_side_results = raytrace_pixels(sdf_network, raytracer, pos_side_uv, camera, bound_r, outer_bound_r,
                                       floor_solver, max_num_rays=max_num_rays, cam_mask=edge_cam_mask,
                                       mock_inner=mock_inner)
    neg_side_results = raytrace_pixels(sdf_network, raytracer, neg_side_uv, camera, bound_r, outer_bound_r,
                                       floor_solver, max_num_rays=max_num_rays, cam_mask=edge_cam_mask,
                                       mock_inner=mock_inner)
    render_normal_and_color(pos_side_results, sdf_network, color_network_dict, renderer, use_white_bkgd,
                            do_reparam=do_reparam, is_training=is_training, max_num_pts=1024 // light_batch_size)
    render_normal_and_color(neg_side_results, sdf_network, color_network_dict, renderer, use_white_bkgd,
                            do_reparam=do_reparam, is_training=is_training, max_num_pts=1024 // light_batch_size)
    # ic(pos_side_results.keys(), pos_side_results['convergent_mask'].sum())

    assign_keys = ['color_fine', 'specular', 'albedo', 'reg_normal', 'visibility', 'inv_s', 'alpha_mask']
    value_ndims = [2, 2, 1, 1, 2, 2, 1]
    for key, value_ndim in zip(assign_keys, value_ndims):
        if key in results:
            this_pos_side_weight = pos_side_weight.view(-1, *[1 for _ in range(value_ndim)])
            results[key].view(-1, *results[key].shape[-value_ndim:])[edge_pixel_idx] = \
                pos_side_results[key] * this_pos_side_weight + \
                neg_side_results[key] * (1.0 - this_pos_side_weight)

    results["edge_pos_neg_camera_points"] = torch.cat(
        [
            pos_side_results["camera_points"][pos_side_results["convergent_mask"]],
            neg_side_results["camera_points"][neg_side_results["convergent_mask"]],
        ],
        dim=0,
    )

    if 'shadow_points' in pos_side_results:
        results["edge_pos_neg_shadow_points"] = torch.cat(
            [
                pos_side_results["shadow_points"][pos_side_results["convergent_mask"]],
                neg_side_results["shadow_points"][neg_side_results["convergent_mask"]],
            ],
            dim=0,
        )

    if 'shadow_error' in pos_side_results:
        results["edge_pos_neg_shadow_error"] = torch.cat(
            [
                pos_side_results["shadow_error"][pos_side_results["convergent_mask"]],
                neg_side_results["shadow_error"][neg_side_results["convergent_mask"]],
            ],
            dim=0,
        )
    # debug
    # results["uv"][edge_mask] = edge_uv.detach()
    # results["points"][edge_mask] = edge_points.detach()

    results["uv"].view(-1, 2)[edge_pixel_idx] = edge_uv.detach()
    results["points"].view(-1, 3)[edge_pixel_idx] = edge_points.detach()

    if VERBOSE_MODE:
        pos_side_weight_fullsize = torch.zeros_like(edge_mask).float()
        # pos_side_weight_fullsize[edge_mask] = pos_side_weight
        pos_side_weight_fullsize.view(-1)[edge_pixel_idx] = pos_side_weight

        pos_side_depth = torch.zeros_like(edge_mask).float()
        # pos_side_depth[edge_mask] = pos_side_results["depth"]
        pos_side_depth.view(-1)[edge_pixel_idx] = pos_side_results["depth"]
        neg_side_depth = torch.zeros_like(edge_mask).float()
        # neg_side_depth[edge_mask] = neg_side_results["depth"]
        neg_side_depth.view(-1)[edge_pixel_idx] = neg_side_results["depth"]

        pos_side_color = (
            torch.zeros(
                list(edge_mask.shape)
                + [
                    3,
                ]
            )
            .float()
            .to(edge_mask.device)
        )
        # pos_side_color[edge_mask] = pos_side_results["color"]
        pos_side_color.view(-1, 3)[edge_pixel_idx] = pos_side_results["color"]
        neg_side_color = (
            torch.zeros(
                list(edge_mask.shape)
                + [
                    3,
                ]
            )
            .float()
            .to(edge_mask.device)
        )
        # neg_side_color[edge_mask] = neg_side_results["color"]
        neg_side_color.view(-1, 3)[edge_pixel_idx] = neg_side_results["color"]
        results.update(
            {
                "edge_pos_side_weight": pos_side_weight_fullsize,
                "edge_normals2d": edge_normals2d,
                "pos_side_uv": pos_side_uv,
                "neg_side_uv": neg_side_uv,
                "edge_pos_side_depth": pos_side_depth,
                "edge_neg_side_depth": neg_side_depth,
                "edge_pos_side_color": pos_side_color,
                "edge_neg_side_color": neg_side_color,
            }
        )


def render_camera(
        camera,
        pixels,
        sdf_network,
        raytracer,
        color_network_dict,
        renderer,
        light_batch_size,
        use_white_bkgd,
        bound_r=1.0,
        outer_bound_r=1.0,
        floor_solver=None,
        handle_edges=True,
        is_training=False,
        full_camera=False,
        use_padded=False,
        do_reparam=True,
        cam_mask=None,
        mock_inner=False,
):
    results = raytrace_camera(
        camera,
        pixels,
        sdf_network,
        raytracer,
        bound_r,
        outer_bound_r,
        floor_solver,
        max_num_rays=1024,
        detect_edges=handle_edges,
        full_camera=full_camera,
        use_padded=use_padded,
        cam_mask=cam_mask,
        mock_inner=mock_inner,
    )
    render_normal_and_color(
        results,
        sdf_network,
        color_network_dict,
        renderer,
        use_white_bkgd,
        do_reparam=do_reparam,
        is_training=is_training,
        max_num_pts=1024 // light_batch_size,
    )
    if handle_edges and results["edge_mask"].sum() > 0:
        render_edge_pixels(
            results,
            camera,
            sdf_network,
            raytracer,
            color_network_dict,
            renderer,
            bound_r,
            outer_bound_r,
            floor_solver,
            use_white_bkgd,
            do_reparam=do_reparam,
            max_num_rays=1024,
            is_training=is_training,
            light_batch_size=light_batch_size,
            cam_mask=cam_mask,
            mock_inner=mock_inner,
        )
    if is_training:
        # camera ray eikonal loss
        eik_points = torch.empty(pixels.numel() // 2, 3).cuda().float().uniform_(-bound_r, bound_r)
        if bound_r != outer_bound_r:
            outer_eik_points = torch.empty(pixels.numel() // 2, 3
                                           ).cuda().float().uniform_(-outer_bound_r, outer_bound_r)
            eik_points = torch.cat([eik_points, outer_eik_points], dim=0)
        mask = results["convergent_mask"]

        def get_extra_points(points):
            extra_points = points.repeat(light_batch_size - 1, 1)
            extra_points += 1e-4 * (torch.rand_like(extra_points) * 2 - 1)
            return extra_points

        extra_eik_error = torch.tensor(0.0)
        extra_eik_cnt = 0
        if mask.any():
            camera_points = results['camera_points'][mask]
            eik_points = torch.cat([eik_points, camera_points], dim=0)
            if light_batch_size > 1:
                eik_points = torch.cat([eik_points, get_extra_points(camera_points)], dim=0)
            if 'shadow_points' in results:
                shadow_points = results['shadow_points'][mask].view(-1, 3)
                eik_points = torch.cat([eik_points, shadow_points], dim=0)
            if 'shadow_error' in results:
                shadow_error = results['shadow_error'][mask].view(-1)
                extra_eik_error += shadow_error.sum()
                extra_eik_cnt += shadow_error.shape[0]
        if 'edge_pos_neg_camera_points' in results:
            eik_points = torch.cat([eik_points, results['edge_pos_neg_camera_points']], dim=0)
            if light_batch_size > 1:
                eik_points = torch.cat([eik_points, get_extra_points(results['edge_pos_neg_camera_points'])], dim=0)
        if 'edge_pos_neg_shadow_points' in results:
            eik_points = torch.cat([eik_points, results['edge_pos_neg_shadow_points'].view(-1, 3)], dim=0)
        if 'edge_pos_neg_shadow_error' in results:
            edge_pos_neg_shadow_error = results['edge_pos_neg_shadow_error'].view(-1)
            extra_eik_error += edge_pos_neg_shadow_error.sum()
            extra_eik_cnt += edge_pos_neg_shadow_error.shape[0]
        if 'edge_points' in results:
            eik_points = torch.cat([eik_points, results['edge_points']], dim=0)
            if light_batch_size > 1:
                eik_points = torch.cat([eik_points, get_extra_points(results['edge_points'])], dim=0)
        pts_norm = torch.linalg.norm(eik_points, ord=2, dim=-1)
        relax_inside_sphere = pts_norm < 1.2 * outer_bound_r
        if relax_inside_sphere.any():
            eik_points = eik_points[relax_inside_sphere]
            eik_grad = sdf_network.gradient(eik_points.detach()).view(-1, 3)
            eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).mean()
            eik_cnt = eik_points.shape[0]
        else:
            eik_loss = torch.tensor(0.0)
            eik_cnt = 0
        if extra_eik_cnt > 0:
            eik_loss = (eik_loss * eik_cnt + extra_eik_error) / (eik_cnt + extra_eik_cnt)
            eik_cnt += extra_eik_cnt
        results.update({
            'gradient_error': eik_loss,
            'eik_cnt': eik_cnt,
        })

    return results
