import mcubes
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.raytracer import intersect_sphere


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 albedo_network,
                 sg_basis,
                 specular_network,
                 dataset,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 obj_near,
                 use_point_light=False,
                 bound_r=2.0,
                 outer_bound_r=None,
                 n_near=0,
                 z_neg=-0.05,
                 ambient_coeff=0.0,
                 floor_solver=None,
                 clip_vis=False,
                 weighted_gradients=False,
                 light_intensity=1.0,
                 mock_phong=False,
                 test_mode=False):
        # self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.albedo_network = albedo_network
        self.sg_basis = sg_basis
        self.specular_network = specular_network
        self.dataset = dataset
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_near = n_near
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.obj_near = obj_near
        self.use_point_light = use_point_light
        self.bound_r = bound_r
        self.outer_bound_r = outer_bound_r if outer_bound_r is not None else bound_r
        self.z_neg = z_neg
        self.albedo_act = nn.Softplus(beta=100)
        self.specular_act = nn.Softplus(beta=100)  # following sdf in IDR
        self.ambient_coeff = ambient_coeff
        self.floor_solver = floor_solver
        self.clip_vis = clip_vis
        self.weighted_gradients = weighted_gradients
        self.light_intensity = light_intensity
        self.mock_phong = mock_phong
        self.test_mode = test_mode
        self.phong_albedo = torch.tensor([0.75, 0.75, 0.75])
        self.phong_specular = torch.tensor(
            [0, 0, 1.5, 0, 0, 0, 0, 0, 0])[None, ...].expand(3, 9).flatten()
        self.floor_phong = True

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < self.outer_bound_r) | (radius[:, 1:] < self.outer_bound_r)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        p = next_sdf - prev_sdf
        c = next_z_vals - prev_z_vals
        ada_eps = torch.maximum(p.detach(), c.detach()) * 1e-5 + np.finfo(np.float32).tiny
        cos_val = p / (c + ada_eps)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        ada_eps = torch.maximum(p.detach(), c.detach()) * 1e-5 + np.finfo(np.float32).tiny
        alpha = (p + ada_eps) / (c + ada_eps)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def ray_marching(self,
                     rays_o,
                     rays_d,
                     z_vals,
                     sample_dist,
                     sdf_network,
                     inv_s,
                     cos_anneal_ratio=0.0,
                     is_training=True):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        inv_s = inv_s.unsqueeze(1).expand(batch_size, n_samples, 1)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        inv_s = inv_s.reshape(-1, 1)

        sdf, _, gradients = sdf_network.get_all(pts, is_training=is_training)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        ada_eps = torch.maximum(p.detach(), c.detach()) * 1e-5 + np.finfo(np.float32).tiny
        alpha = ((p + ada_eps) / (c + ada_eps)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False).reshape(alpha.shape)
        inside_sphere = (radius < self.outer_bound_r).float()
        alpha = alpha * inside_sphere

        transmittence = torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights = alpha * transmittence
        weights_sum = weights.sum(dim=-1, keepdim=True)

        res_dict = {
            'pts': pts,
            'weights': weights,
            'weights_sum': weights_sum,
        }
        return res_dict

    @torch.no_grad()
    def backward_sampling(self, rays_o, rays_d, normals_o, near, far, n_samples_back, perturb_overwrite=-1):
        batch_size = len(rays_o)
        z_vals = self.uniform_back_sample(far, n_samples_back, near, perturb_overwrite)

        neg_mask = (normals_o * rays_d).sum(dim=-1, keepdim=True) < 0
        sign = torch.ones_like(neg_mask, dtype=torch.float32)
        sign[neg_mask] = -1

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.zeros_like(dists[..., -1:])], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples_back)
        sdf *= sign
        sdf_diff = sdf[..., 1:] - sdf[..., :-1]
        sdf_diff = (sdf_diff >= 0).float()
        sdf_diff = torch.flip(sdf_diff, dims=[-1])
        # sdf_diff = torch.cat([torch.ones_like(sdf_diff[..., :1]), sdf_diff], dim=-1)
        sdf_diff = torch.cat([sdf_diff, torch.zeros_like(sdf_diff[..., -1:])], dim=-1)
        sdf_diff = torch.cumprod(sdf_diff, dim=-1)
        sdf_diff_val = sdf_diff * torch.arange(1, sdf_diff.shape[-1] + 1)[None, ...]
        # sdf_diff = torch.flip(sdf_diff, dims=[-1])
        sdf_diff_val = torch.flip(sdf_diff_val, dims=[-1])
        max_indices = torch.argmax(sdf_diff_val, dim=-1, keepdim=True)  # first valid value
        # retrive first valid z_val
        first_z_vals = torch.gather(z_vals, -1, max_indices)
        # sdf_diff = (sdf_diff > 0.5).float()
        # z_vals = z_vals * sdf_diff + first_z_vals * (1 - sdf_diff)

        # # debug current sdf
        # pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        # sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples_back)
        # sdf *= sign
        # sdf_diff = sdf[..., 1:] - sdf[..., :-1]
        # if torch.any(sdf_diff < 0):
        # import ipdb; ipdb.set_trace()
        # pass
        # resample
        z_vals = self.uniform_back_sample(far, n_samples_back, first_z_vals, perturb_overwrite)
        return {
            'z_vals': z_vals,
            'n_samples': n_samples_back,
        }

    def uniform_back_sample(self, far, n_samples_back, near, perturb_overwrite):
        z_vals = torch.linspace(0.0, 1.0, n_samples_back + 1)
        z_vals = near + (far - near) * z_vals[None, :]
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
        z_vals = z_vals[..., :-1]
        return z_vals

    @torch.no_grad()
    def importance_sampling(self, rays_o, rays_d, near, far, perturb_overwrite=-1, back_z_dict=None):
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        # Up sample
        if self.n_importance > 0 or back_z_dict is not None:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, -1)

                if back_z_dict is not None:
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  back_z_dict['z_vals'],
                                                  sdf,
                                                  last=False)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance
            if back_z_dict is not None:
                n_samples += back_z_dict['n_samples']

        return {
            'n_samples': n_samples,
            'z_vals': z_vals,
        }

    def render_fn(self, interior_mask, sdf_network, color_network_dict, ray_o, ray_d, obj_o, grads, features,
                  is_training,
                  use_white_bkgd, sample_dist=1e10):
        light_bs = color_network_dict['lights_o'].shape[0]

        color_fine = None
        specular = None
        albedo = None
        normal = None
        visibility = None
        camera_points = None
        depth_points = None
        shadow_error = None
        inv_s = None

        if interior_mask.any():
            camera_points = obj_o.detach()
            # light batch
            ray_bs = ray_o.shape[0]

            def expand_shape(x, dim):
                return x.unsqueeze(dim).expand(ray_bs, light_bs, *x.shape[1:]).reshape(-1, *x.shape[1:])

            ray_d_expanded = expand_shape(ray_d, 1)
            obj_o_expanded = expand_shape(obj_o, 1)
            lights_o = expand_shape(color_network_dict['lights_o'], 0)
            lights_d = expand_shape(color_network_dict['lights_d'], 0)

            # light direction
            if self.use_point_light:
                light_obj_d = lights_o - obj_o_expanded
                light_obj_d_norm = torch.norm(light_obj_d, dim=-1, keepdim=True)
                light_obj_d = light_obj_d / light_obj_d_norm
                falloff = 1 / light_obj_d_norm ** 2
            else:
                light_obj_d = -lights_d
                falloff = 1.

            # shadow ray range
            inner_mask_intersect, inner_z_near, inner_z_far = intersect_sphere(obj_o_expanded, light_obj_d,
                                                                               r=self.bound_r, keepdim=True)
            _, outer_z_near, outer_z_far = intersect_sphere(obj_o_expanded, light_obj_d,
                                                            r=self.outer_bound_r, keepdim=True)
            outer_z_near.clamp_(min=self.z_neg)
            inner_z_near.clamp_(min=self.obj_near)
            no_inner_mask = ~inner_mask_intersect
            inner_z_near[no_inner_mask] = outer_z_near[no_inner_mask] + (
                    outer_z_far[no_inner_mask] - outer_z_near[no_inner_mask]) * self.n_near / (
                                                  self.n_near + self.n_samples)
            inner_z_far[no_inner_mask] = outer_z_far[no_inner_mask]
            inner_z_far = torch.maximum(inner_z_far, inner_z_near + 0.5)  # prevent volume rendering in small ranges
            outer_z_near = torch.minimum(outer_z_near, inner_z_near - 0.01)

            normal = torch.nn.functional.normalize(grads, dim=-1)
            normal_expanded = expand_shape(normal, 1)

            # samples near start surface
            if self.n_near > 0:
                obj_back_dict = self.backward_sampling(obj_o_expanded, light_obj_d, normal_expanded,
                                                       outer_z_near, inner_z_near,
                                                       self.n_near,
                                                       perturb_overwrite=color_network_dict['perturb_overwrite'])
            else:
                obj_back_dict = None

            inv_s = self.deviation_network(torch.zeros([obj_o_expanded.shape[0], 3]))
            # Single parameter
            inv_s = inv_s[..., :1].clip(1e-6, 1e6)
            if color_network_dict.get('render_visibility', True):
                # visibility
                obj_z_dict = self.importance_sampling(obj_o_expanded, light_obj_d, inner_z_near, inner_z_far,
                                                      perturb_overwrite=color_network_dict['perturb_overwrite'],
                                                      back_z_dict=obj_back_dict)
                obj_ray_dict = self.ray_marching(obj_o_expanded, light_obj_d, obj_z_dict['z_vals'], sample_dist,
                                                 self.sdf_network, inv_s,
                                                 cos_anneal_ratio=color_network_dict['cos_anneal_ratio'],
                                                 is_training=is_training)
                visibility = 1.0 - obj_ray_dict['weights_sum']  # batch_size, 1
                if self.weighted_gradients:
                    gradients = self.sdf_network.gradient(obj_ray_dict['pts'].detach()).view(-1, 3)
                    gradient_error = ((gradients.norm(dim=-1) - 1.0) ** 2).view(obj_ray_dict['weights'].shape)
                    shadow_error = torch.sum(gradient_error * obj_ray_dict['weights'].detach(), dim=-1,
                                             keepdim=True) / (obj_ray_dict['weights_sum'].detach() + 1e-10)
                else:
                    with torch.no_grad():  # Eikonal loss sample points
                        depth = torch.sum(obj_z_dict['z_vals'] * obj_ray_dict['weights'], dim=-1, keepdim=True)
                        depth = depth / (obj_ray_dict['weights_sum'] + 1e-10)
                        depth_points = obj_o_expanded + light_obj_d * depth
                if self.clip_vis:
                    visibility = visibility.clip(1e-3, 1.0 - 1e-3)
            else:
                visibility = torch.ones_like(obj_o_expanded[..., :1])  # mock visibility
                depth_points = obj_o_expanded.detach()

            # color
            if color_network_dict['render_color']:
                screen_feat = self.albedo_network(obj_o, normal, None, features)
                albedo = self.albedo_act(screen_feat[..., 0:3])
                if self.floor_phong or self.floor_solver is None:
                    phong_mask = torch.ones_like(obj_o[..., 0], dtype=torch.bool)
                else:
                    phong_mask = (obj_o[..., 2] > self.floor_solver.floor_height + 0.01) & (obj_o.norm(
                        dim=-1) < self.bound_r * 0.75)
                if self.mock_phong:
                    albedo[phong_mask] = self.phong_albedo
                if self.sg_basis is not None:  # have specular
                    if self.specular_network is not None:
                        specular_feat = self.specular_network(obj_o, normal, None, features)
                    else:
                        specular_feat = screen_feat[..., 3:3 + self.sg_basis.nchannel * self.sg_basis.nbasis]
                    specular_weights = self.specular_act(specular_feat)
                    if self.mock_phong:
                        specular_weights[phong_mask] = self.phong_specular
                    specular_weights_expanded = expand_shape(specular_weights, 1)
                    specular_n = normal_expanded
                    specular_l = light_obj_d
                    specular_v = -ray_d_expanded
                    specular = self.sg_basis(l=specular_l, v=specular_v, n=specular_n,
                                             weights=specular_weights_expanded)
                else:
                    specular = expand_shape(torch.zeros_like(albedo), 1)

                albedo_expanded = expand_shape(albedo, 1)

                color_fine = self.specular_shading(albedo_expanded, specular, visibility, light_obj_d,
                                                   normal_expanded, falloff)

        def complete_mask(x, tail_shape, _use_white_bkgd):
            dots_sh = tuple(interior_mask.shape)
            ret_x = torch.full(
                dots_sh + tail_shape,
                fill_value=1.0 if _use_white_bkgd else 0.0,
                dtype=torch.float32,
                device=interior_mask.device,
            )
            if x is not None:
                ret_x[interior_mask] = x.view(-1, *tail_shape)
            return ret_x

        # use interior_mask to get complete return
        res_dict = {
            'visibility': complete_mask(visibility, (light_bs, 1), use_white_bkgd),
            'camera_points': complete_mask(camera_points, (3,), False),
            'reg_normal': complete_mask(normal, (3,), False),
            'inv_s': complete_mask(inv_s, (light_bs, 1), True),
            'alpha_mask': interior_mask.unsqueeze(-1).float(),
        }
        if self.weighted_gradients:
            res_dict.update({
                'shadow_error': complete_mask(shadow_error, (light_bs, 1), False)
            })
        else:
            res_dict.update({
                'shadow_points': complete_mask(depth_points, (light_bs, 3), False),
            })
        if color_network_dict['render_color']:
            res_dict.update({
                'color_fine': complete_mask(color_fine, (light_bs, 3), use_white_bkgd),
                'specular': complete_mask(specular, (light_bs, 3,), use_white_bkgd),
                'albedo': complete_mask(albedo, (3,), use_white_bkgd),
            })
        return res_dict

    def specular_shading(self, albedo, specular, visibility, ray_light, normal, falloff):
        shading = torch.sum(normal * ray_light, dim=-1, keepdim=True)
        color_shading = shading * (albedo + specular) * falloff * self.light_intensity

        neg_shading_mask = shading[..., 0] <= 0.0
        color_shading = color_shading.clone()
        color_shading[neg_shading_mask] = color_shading[neg_shading_mask].detach()

        if self.test_mode:
            color_shading = color_shading.clamp(min=0.0)

        res = color_shading * visibility + self.ambient_coeff * albedo
        return res

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):

        def clipped_sdf(pts):
            res = -self.sdf_network.sdf(pts)
            if self.floor_solver is not None:
                up_floor = pts[..., 2] >= self.floor_solver.floor_height - 0.05
                res[~up_floor] = -1e9
            return res

        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=clipped_sdf)
