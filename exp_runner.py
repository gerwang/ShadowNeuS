import argparse
import json
import logging
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from glob import glob
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import nerf_dataset, deep_shadow_dataset, real_dataset
from models.base_dataset import linear_to_srgb, euler_radius_to_pose, pose_to_euler_radius, srgb_to_linear
from models.depth_normal import get_depth_map_faces, norm_diff
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from models.floor import FloorHeight, DepthFloorHeight
from models.raytracer import RayTracer, render_camera
from models.renderer import NeuSRenderer
from models.resolution_scheduler import ResolutionScheduler
from models.sgbasis import SGBasis


class Runner:
    def __init__(self, conf_path, **kwargs):
        self.device = torch.device('cuda')
        mode = kwargs.get('mode', 'train')
        case = kwargs.get('case', 'CASE_NAME')
        is_continue = kwargs.get('is_continue', False)
        split = kwargs.get('split', None)
        data_sub = kwargs.get('data_sub', None)
        suffix = kwargs.get('suffix', '')
        load_scene = kwargs.get('load_scene', False)
        self.test_mode = kwargs.get('test_mode', False)
        if suffix != '':
            suffix = '_' + suffix

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        if split is not None:
            self.conf.put('dataset.split', split)
        if data_sub is not None:
            self.conf.put('dataset.data_sub', data_sub)
        if mode == 'train' and self.conf.get_float('train.contain_weight', 0.0) > 0 \
                and 'dataset.dilate' not in self.conf:
            self.conf.put('dataset.dilate', True)  # corrupt the mask to a rough bounding box of the object
        self.base_exp_dir = self.conf['general.base_exp_dir'] + suffix
        os.makedirs(self.base_exp_dir, exist_ok=True)
        if 'DeepShadowData' in self.conf['dataset.data_dir']:
            self.dataset = deep_shadow_dataset.DeepShadowDataset(self.conf['dataset'])
        elif 'real_data' in self.conf['dataset.data_dir']:
            self.dataset = real_dataset.RealDataset(self.conf['dataset'])
        else:
            self.dataset = nerf_dataset.NeRFDataset(self.conf['dataset'])
        self.iter_step = 0

        if 'resolution_scheduler' in self.conf['model']:
            self.resolution_scheduler = ResolutionScheduler(**self.conf['model.resolution_scheduler'])
        else:
            self.resolution_scheduler = None

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.light_batch_size = self.conf.get_int('train.light_batch_size', 1)
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        if self.test_mode:
            self.use_white_bkgd = True
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.mock_inner_end = self.conf.get_int('train.mock_inner_end', default=0)

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight', 1.0)
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.vis_weight = self.conf.get_float('train.vis_weight', 0.0)
        self.bound_weight = self.conf.get_float('train.bound_weight', 0.0)
        self.contain_weight = self.conf.get_float('train.contain_weight', 0.0)
        self.vis_loss_type = self.conf.get('train.vis_loss_type', 'bce')
        self.color_loss_type = self.conf.get('train.color_loss_type', 'l1')
        self.anneal_fore_iter = self.conf.get_int('train.anneal_fore_iter', -1)
        self.anneal_fore_alpha = self.conf.get_float('train.anneal_fore_alpha', 1.0)
        self.anneal_back_alpha = self.conf.get_float('train.anneal_back_alpha', 1.0)
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.albedo_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        if 'model.specular_network' in self.conf:
            self.specular_network = RenderingNetwork(**self.conf['model.specular_network']).to(self.device)
        else:
            self.specular_network = None
        if 'model.sgbasis' in self.conf:
            self.sg_basis = SGBasis(**self.conf['model.sgbasis'])
        else:
            self.sg_basis = None

        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.albedo_network.parameters())
        if self.specular_network is not None:
            params_to_train += list(self.specular_network.parameters())
        if self.sg_basis is not None:
            params_to_train += list(self.sg_basis.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        up_floor = self.conf.get_bool('train.up_floor', False)
        floor_type = self.conf.get_string('train.floor_type', 'plane')
        if up_floor:
            if isinstance(self.dataset, nerf_dataset.NeRFDataset) or \
                    isinstance(self.dataset, deep_shadow_dataset.DeepShadowDataset) or \
                    isinstance(self.dataset, real_dataset.RealDataset):
                floor_height = self.dataset.floor_height
            else:
                raise ValueError(f'dataset {type(self.dataset)} does not support up_floor')
            if floor_type == 'depth':
                self.floor_solver = DepthFloorHeight(floor_height,
                                                     self.dataset.get_camera(0, 1),
                                                     glob(f'{self.dataset.data_dir}/floor/*_depth_*.exr')[0],
                                                     self.device)
            elif floor_type == 'plane':
                self.floor_solver = FloorHeight(floor_height)
            else:
                raise ValueError(f'unknown {floor_type}')
        else:
            self.floor_solver = None

        self.iron_tracer = RayTracer(**self.conf.get('model.ray_tracer', {}), test_mode=self.test_mode)

        self.renderer = NeuSRenderer(self.sdf_network, self.deviation_network, self.albedo_network, self.sg_basis,
                                     self.specular_network, self.dataset,
                                     **self.conf['model.neus_renderer'], floor_solver=self.floor_solver,
                                     test_mode=self.test_mode)

        # Load checkpoint
        latest_model_name = None
        if is_continue or load_scene:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            if is_continue:
                self.load_checkpoint(latest_model_name)
            else:
                self.load_checkpoint_scene_only(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            resolution_level = 1 if self.resolution_scheduler is None else self.resolution_scheduler.update_res_level(
                self.iter_step)
            image_idx = image_perm[self.iter_step % len(image_perm)]
            view_idx, light_idx = self.dataset.get_view_light_idx(image_idx)
            batch_img_idx = torch.tensor(
                [image_idx, *self.dataset.random_image_idx_same_view(
                    view_idx, self.light_batch_size - 1, exclude_idx=image_idx)],
                dtype=torch.long)  # unique image indexes
            batch_pixels = self.dataset.gen_random_pixels_x_pixels_y(
                self.batch_size, resolution_level)  # unique pixel indexes
            lights_o, lights_d = self.dataset.get_light_center_and_ray(batch_img_idx)
            camera = self.dataset.get_camera(image_idx, resolution_level)
            camera_ray_data = self.dataset.gen_camera_ray_data(batch_pixels, image_idx, resolution_level)
            shadow_ray_data = self.dataset.gen_shadow_ray_data(batch_pixels, batch_img_idx, resolution_level)
            if 'mask' in camera_ray_data:
                cam_mask = camera_ray_data['mask'][..., 0]
            else:
                cam_mask = None
            render_out = render_camera(
                camera,
                batch_pixels,
                self.sdf_network,
                self.iron_tracer,
                {
                    'lights_o': lights_o,
                    'lights_d': lights_d,
                    'perturb_overwrite': -1,
                    'cos_anneal_ratio': self.get_cos_anneal_ratio(),
                    'render_color': self.color_weight > 0,
                    'render_visibility': True,
                },
                self.renderer,
                light_batch_size=self.light_batch_size,
                use_white_bkgd=self.use_white_bkgd,
                bound_r=self.renderer.bound_r,
                outer_bound_r=self.renderer.outer_bound_r,
                floor_solver=self.floor_solver,
                is_training=True,
                full_camera=False,
                use_padded=True,
                cam_mask=cam_mask.bool(),
                mock_inner=self.iter_step < self.mock_inner_end,
            )
            # Loss
            if self.color_weight > 0:
                color_fine = render_out['color_fine']
                true_rgb = shadow_ray_data['color']
                color_error = color_fine - true_rgb
                if self.anneal_fore_iter > 0 and self.iter_step < self.anneal_fore_iter:
                    # When training starts, rely more on shadows cast on background
                    color_mask = cam_mask
                    color_mask = self.anneal_fore_alpha * color_mask + self.anneal_back_alpha * (1.0 - color_mask)
                    color_mask = color_mask[..., None, None]
                    color_error = color_error * color_mask
                else:
                    color_mask = torch.ones_like(shadow_ray_data['color'][..., :1])
                color_mask_sum = color_mask.sum() + 1e-5
                if self.color_loss_type == 'l1':
                    color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),
                                                reduction='sum') / color_mask_sum
                elif self.color_loss_type == 'smooth_l1':
                    color_fine_loss = F.smooth_l1_loss(color_error, torch.zeros_like(color_error), beta=0.1,
                                                       reduction='sum') / color_mask_sum
                else:
                    raise ValueError(f'unknown {self.color_loss_type}')
                psnr = 20.0 * torch.log10(
                    1.0 / (((color_fine - true_rgb) ** 2 * color_mask).sum() / (color_mask_sum * 3.0)).sqrt())
            else:
                color_fine_loss = torch.tensor(0.0)
                psnr = 0.0

            extra_eik_points = []

            bound_loss = torch.tensor(0.0)
            if self.bound_weight > 0:  # guard the sdf to stay in the bounding sphere
                bound_points = torch.empty(self.batch_size * self.light_batch_size, 3).cuda().float().uniform_(-1, 1)
                bound_points = F.normalize(bound_points, dim=-1) * self.renderer.bound_r * torch.distributions.Uniform(
                    0.99, 1.03).sample(bound_points.shape[:-1])[..., None]
                sdf_output = self.sdf_network.sdf(bound_points)
                valid_mask = sdf_output[..., 0] <= 0
                if self.floor_solver is not None:
                    valid_mask = valid_mask & (bound_points[..., 2] >= self.floor_solver.floor_height)
                if valid_mask.any():
                    valid_sdf_output = sdf_output[valid_mask]
                    bound_loss = F.smooth_l1_loss(valid_sdf_output, torch.zeros_like(valid_sdf_output), beta=0.1)
                    extra_eik_points.append(bound_points[valid_mask])

            if self.vis_weight > 0:
                visibility = render_out['visibility']
                vis = shadow_ray_data['vis']
                if self.anneal_fore_iter > 0 and self.iter_step < self.anneal_fore_iter:
                    # When training starts, rely more on shadows cast on background
                    mask = cam_mask
                    mask = self.anneal_fore_alpha * mask + self.anneal_back_alpha * (1.0 - mask)
                    mask = mask[..., None, None]
                    vis = vis * mask
                    visibility = visibility * mask
                if self.vis_loss_type == 'bce':
                    vis_loss = F.binary_cross_entropy(visibility.clip(1e-3, 1.0 - 1e-3), vis)
                elif self.vis_loss_type == 'l1':
                    vis_loss = F.l1_loss(visibility, vis)
                elif self.vis_loss_type == 'smooth_l1':
                    vis_loss = F.smooth_l1_loss(visibility, vis, beta=0.1)
                else:
                    raise ValueError(f'unknown {self.vis_loss_type}')
            else:
                vis_loss = torch.tensor(0.0)

            contain_loss = torch.tensor(0.0)
            if self.contain_weight > 0:
                floor_rays_o, floor_rays_d, _ = camera.get_rays(render_out['uv'])
                floor_distance = self.floor_solver.get_floor_distance(floor_rays_o, floor_rays_d)
                floor_points = floor_rays_o + floor_distance * floor_rays_d
                relax_inside_sphere = floor_points.norm(dim=-1) < self.renderer.bound_r * 1.2
                mask = cam_mask.bool()  # a coarse bounding box of the object
                non_mask = ~mask
                non_real_mask = non_mask & render_out['real_mask']
                non_contain_mask = relax_inside_sphere & non_mask
                layer_thickness = 0.05 * self.renderer.bound_r
                if non_contain_mask.any():  # regularize surface to stay at the floor
                    non_contain_points = floor_points[non_contain_mask]
                    sdf_output = self.sdf_network.sdf(non_contain_points)
                    contain_loss = F.smooth_l1_loss(sdf_output, torch.zeros_like(
                        sdf_output), beta=0.1)  # We want floor to be on the zero level set
                    extra_eik_points.append(non_contain_points)

                if non_real_mask.any():  # remove floaters in front of the floor
                    real_distance = ((render_out['real_points'] - floor_rays_o)
                                     * floor_rays_d).sum(dim=-1, keepdims=True)
                    upper_distance_lo = torch.minimum(floor_distance - layer_thickness, real_distance)
                    upper_distance_hi = torch.minimum(floor_distance, real_distance + layer_thickness)
                    upper_distance = upper_distance_lo + torch.rand_like(upper_distance_lo) * (
                            upper_distance_hi - upper_distance_lo)
                    upper_points = (floor_rays_o + upper_distance * floor_rays_d)[non_real_mask]
                    upper_sdf_output = self.sdf_network.sdf(upper_points)
                    upper_sdf_neg_mask = upper_sdf_output < 0
                    if upper_sdf_neg_mask.any():
                        neg_upper_sdf_output = upper_sdf_output[upper_sdf_neg_mask]
                        contain_loss += F.smooth_l1_loss(neg_upper_sdf_output, torch.zeros_like(neg_upper_sdf_output),
                                                         beta=0.1)
                        extra_eik_points.append(upper_points[upper_sdf_neg_mask[..., 0]])

                if relax_inside_sphere.any():  # add thickness to the floor, as ray marching sometimes overlook thin surface
                    inside_floor_points = floor_points[relax_inside_sphere]
                    under_points = inside_floor_points - torch.tensor([0, 0, layer_thickness])[
                        None, ...] * (1.0 - torch.rand(*inside_floor_points.shape[:-1], 1))
                    under_sdf_output = self.sdf_network.sdf(under_points)
                    under_sdf_pos_mask = under_sdf_output > 0
                    if under_sdf_pos_mask.any():
                        pos_under_sdf_output = under_sdf_output[under_sdf_pos_mask]
                        contain_loss += F.smooth_l1_loss(pos_under_sdf_output, torch.zeros_like(pos_under_sdf_output),
                                                         beta=0.1)
                        extra_eik_points.append(under_points[under_sdf_pos_mask[..., 0]])

            eikonal_loss = render_out['gradient_error']
            if len(extra_eik_points) > 0:
                extra_eik_points = torch.cat(extra_eik_points, dim=0)
                extra_eik_grad = self.sdf_network.gradient(extra_eik_points.detach()).view(-1, 3)
                extra_eik_cnt = extra_eik_points.shape[0]
                extra_eik_loss = ((extra_eik_grad.norm(dim=-1) - 1) ** 2).mean()
                eikonal_loss = (eikonal_loss * render_out['eik_cnt'] + extra_eik_loss * extra_eik_cnt) / (
                        render_out['eik_cnt'] + extra_eik_cnt)

            loss = (color_fine_loss * self.color_weight + eikonal_loss * self.igr_weight +
                    vis_loss * self.vis_weight +
                    bound_loss * self.bound_weight + contain_loss * self.contain_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            if self.color_weight > 0:
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            if self.igr_weight > 0:
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            if self.vis_weight > 0:
                self.writer.add_scalar('Loss/vis_loss', vis_loss, self.iter_step)
            if self.bound_weight > 0:
                self.writer.add_scalar('Loss/bound_loss', bound_loss, self.iter_step)
            if self.contain_weight > 0:
                self.writer.add_scalar('Loss/contain_loss', contain_loss, self.iter_step)

            if self.color_weight > 0:
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            if self.resolution_scheduler is not None:
                self.writer.add_scalar('Statistics/resolution_level', resolution_level, self.iter_step)
            if 'edge_mask' in render_out:
                self.writer.add_scalar('Statistics/edge_point_cnt', render_out['edge_mask'].sum(), self.iter_step)
            s_val_mask = render_out['convergent_mask']
            if 'edge_mask' in render_out:
                s_val_mask = s_val_mask | render_out['edge_mask']
            if s_val_mask.any():
                s_val = (1.0 / render_out['inv_s'][s_val_mask]).mean()
            else:
                s_val = torch.tensor(0.0)
            self.writer.add_scalar('Statistics/s_val', s_val, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return np.random.permutation(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.albedo_network.load_state_dict(checkpoint['albedo_network'])
        if not self.test_mode:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        if self.sg_basis is not None:
            self.sg_basis.load_state_dict(checkpoint['sg_basis'])
        if self.specular_network is not None:
            self.specular_network.load_state_dict(checkpoint['specular_network'])

        logging.info('End')

    def load_checkpoint_scene_only(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.albedo_network.load_state_dict(checkpoint['albedo_network'])
        if self.sg_basis is not None:
            self.sg_basis.load_state_dict(checkpoint['sg_basis'])
        if self.specular_network is not None:
            self.specular_network.load_state_dict(checkpoint['specular_network'])

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'albedo_network': self.albedo_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        if self.sg_basis is not None:
            checkpoint.update({
                'sg_basis': self.sg_basis.state_dict(),
            })
        if self.specular_network is not None:
            checkpoint.update({
                'specular_network': self.specular_network.state_dict(),
            })

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_normal_depth(self, idx=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate normal & depth: iter: {}, camera: {}'.format(self.iter_step, idx))

        resolution_level = 1

        camera = self.dataset.get_camera(idx, resolution_level)
        H, W = camera.H, camera.W
        batch_pixels = camera.get_pixels()
        lights_o, lights_d = self.dataset.get_light_center_and_ray(idx)
        camera_ray_data = self.dataset.gen_camera_ray_data(batch_pixels.view(-1, 2), idx, resolution_level)
        if 'mask' in camera_ray_data:
            cam_mask = camera_ray_data['mask'].view(*batch_pixels.shape[:-1])
        else:
            cam_mask = None
        render_out = render_camera(
            camera,
            batch_pixels,
            self.sdf_network,
            self.iron_tracer,
            {
                'lights_o': lights_o,
                'lights_d': lights_d,
                'perturb_overwrite': -1,
                'cos_anneal_ratio': self.get_cos_anneal_ratio(),
                'render_color': self.color_weight > 0,
                'render_visibility': False,
            },
            self.renderer,
            light_batch_size=1,
            use_white_bkgd=self.use_white_bkgd,
            bound_r=self.renderer.bound_r,
            outer_bound_r=self.renderer.outer_bound_r,
            floor_solver=self.floor_solver,
            handle_edges=False,
            is_training=False,
            full_camera=True,
            cam_mask=cam_mask.bool(),
            mock_inner=self.iter_step < self.mock_inner_end,
        )
        os.makedirs(os.path.join(self.base_exp_dir, 'quantitative_compare'), exist_ok=True)

        ray_o, ray_d, ray_d_norm = camera.get_rays(batch_pixels.float() + 0.5)

        world_normal = render_out['reg_normal']
        camera_normal = torch.einsum('ij,NMj->NMi', self.dataset.pose_all_inv[idx, :3, :3], world_normal)
        camera_normal[..., [1, 2]] = -camera_normal[..., [1, 2]]
        normal_img = camera_normal.detach().cpu().numpy()
        depth_map = (((render_out['points'] - ray_o) * ray_d).sum(dim=-1) / ray_d_norm).detach().cpu().numpy()
        normal_gt = self.dataset.normal_gt.detach().cpu().numpy()
        depth_gt = self.dataset.depth_gt.detach().cpu().numpy()
        cv.imwrite(os.path.join(self.base_exp_dir, 'quantitative_compare', 'normal_gt.png'),
                   (normal_gt * 0.5 + 0.5).clip(0, 1)[..., ::-1] * 255)
        cv.imwrite(os.path.join(self.base_exp_dir, 'quantitative_compare', 'normal_pred.png'),
                   (normal_img * 0.5 + 0.5).clip(0, 1)[..., ::-1] * 255)
        np.save(os.path.join(self.base_exp_dir, 'quantitative_compare', 'normal.npy'),
                world_normal.detach().cpu().numpy())
        np.save(os.path.join(self.base_exp_dir, 'quantitative_compare', 'depth.npy'),
                depth_map * ray_d_norm.detach().cpu().numpy())

        def save_verts(verts, name):
            verts_map = verts.reshape(H, W, 3)
            faces = get_depth_map_faces(verts_map, 1e9)
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
                os.path.join(self.base_exp_dir, 'quantitative_compare', f'{name}.ply'))

        ray_d *= ray_d_norm[..., None]
        ray_o = ray_o.detach().cpu().numpy()
        ray_d = ray_d.detach().cpu().numpy()

        obj_pred = ray_o + depth_map[..., None] * ray_d
        save_verts(obj_pred.reshape(-1, 3), 'depth_pred')

        obj_gt = ray_o + depth_gt[..., None] * ray_d
        save_verts(obj_gt.reshape(-1, 3), 'depth_gt')

        normed_pred_depth_map = depth_map - depth_map.min()
        normed_pred_depth_map /= normed_pred_depth_map.max()

        normed_gt_depth_map = depth_gt - depth_gt.min()
        normed_gt_depth_map /= normed_gt_depth_map.max()

        normed_depth_abs_err = np.abs(normed_pred_depth_map - normed_gt_depth_map).mean()

        mask_gt = (self.dataset.masks[resolution_level][idx, ..., 0].to(self.device) > 0.5).float()
        angular_error, angular_error_mean = norm_diff(
            camera_normal.permute(2, 0, 1).unsqueeze(0),
            self.dataset.normal_gt.permute(2, 0, 1).unsqueeze(0),
            mask_gt)

        np_mask_gt = mask_gt.bool().cpu().numpy()
        abs_depth_error = np.abs(depth_gt[np_mask_gt] - depth_map[np_mask_gt]).mean()

        res_dict = {
            'depth_error': float(normed_depth_abs_err),
            'abs_depth_error': float(abs_depth_error),
            'normal_error': float(angular_error_mean),
            'iter_step': self.iter_step,
        }
        json.dump(res_dict, open(os.path.join(self.base_exp_dir, 'quantitative_compare', 'result.json'), 'w'))
        print(res_dict)

    def validate_image(self, idx=-1, resolution_level=-1, s_val=None, save_view=True, base_dir=None,
                       camera=None, idx_name=None, render_visibility=True, light_pose=None, save_exr=False):
        if base_dir is None:
            base_dir = self.base_exp_dir
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if s_val is not None:  # mock s_val
            self.deviation_network.variance.data[...] = -np.log(s_val) / 10  # inv_s
        if idx_name is None:
            if s_val is not None:  # mock s_val
                idx_name = f'{idx}_s_val_{s_val}'
            else:
                idx_name = f'{idx}'
                # print(f's_val: {1 / np.exp(10 * self.deviation_network.variance.item())}')

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        if camera is None:
            camera = self.dataset.get_camera(idx, resolution_level)
        H, W = camera.H, camera.W
        batch_pixels = camera.get_pixels()
        lights_o, lights_d = self.dataset.get_light_center_and_ray(idx, light_pose=light_pose)
        camera_ray_data = self.dataset.gen_camera_ray_data(batch_pixels.view(-1, 2), idx, resolution_level)
        if 'mask' in camera_ray_data:
            cam_mask = camera_ray_data['mask'].view(*batch_pixels.shape[:-1])
        else:
            cam_mask = None
        render_out = render_camera(
            camera,
            batch_pixels,
            self.sdf_network,
            self.iron_tracer,
            {
                'lights_o': lights_o,
                'lights_d': lights_d,
                'perturb_overwrite': 0,
                'cos_anneal_ratio': self.get_cos_anneal_ratio(),
                'render_color': self.color_weight > 0,
                'render_visibility': render_visibility,
            },
            self.renderer,
            light_batch_size=1,
            use_white_bkgd=self.use_white_bkgd,
            bound_r=self.renderer.bound_r,
            outer_bound_r=self.renderer.outer_bound_r,
            floor_solver=self.floor_solver,
            handle_edges=True,
            is_training=False,
            full_camera=True,
            cam_mask=cam_mask.bool(),
            mock_inner=self.iter_step < self.mock_inner_end,
        )
        world_normal = render_out['reg_normal']
        camera_normal = torch.einsum('ij,NMj->NMi', self.dataset.pose_all_inv[idx, :3, :3], world_normal)
        camera_normal[..., [1, 2]] = -camera_normal[..., [1, 2]]
        normal_img = camera_normal.detach().cpu().numpy() * 255
        visibility_img = render_out['visibility'].detach().cpu().numpy()
        visibility_img = (visibility_img[:, :, 0] * 256).clip(0, 255)
        obj_o_verts = render_out['points'].detach().cpu().numpy().reshape(-1, 3)
        alpha_img = render_out['alpha_mask'].detach().cpu().numpy()
        alpha_img = (alpha_img * 256).clip(0, 255)
        os.makedirs(os.path.join(base_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'visibility'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'point_cloud'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'alpha'), exist_ok=True)
        verts_map = obj_o_verts.reshape(H, W, 3)
        faces = get_depth_map_faces(verts_map, 1e9)
        i = 0
        if save_view:
            trimesh.Trimesh(vertices=obj_o_verts, faces=faces, process=False).export(
                os.path.join(base_dir, 'point_cloud', '{:0>8d}_{}_{}.ply'.format(self.iter_step, i, idx_name)))
            cv.imwrite(os.path.join(base_dir,
                                    'normals',
                                    '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                       normal_img[..., ::-1])
        if self.dataset.load_vis and not self.test_mode:
            this_visibility = np.concatenate([visibility_img,
                                              self.dataset.visibility_at(idx,
                                                                         resolution_level=resolution_level)])
        else:
            this_visibility = visibility_img
        cv.imwrite(os.path.join(base_dir,
                                'visibility', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                   this_visibility)
        if save_view:
            cv.imwrite(os.path.join(base_dir,
                                    'alpha', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                       alpha_img)
        if self.color_weight > 0:
            albedo_img = render_out['albedo'].detach().cpu().numpy()
            if self.dataset.convert_linear:
                albedo_img = linear_to_srgb(albedo_img)
            albedo_img = (albedo_img * 256).clip(0, 255)
            specular_img = render_out['specular'].detach().cpu().numpy()
            if self.dataset.convert_linear:
                specular_img = linear_to_srgb(specular_img)
            specular_img = (specular_img[:, :, 0] * 256).clip(0, 255)
            os.makedirs(os.path.join(base_dir, 'validations_fine'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'albedo'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'specular'), exist_ok=True)
            if save_exr:
                img_fine = render_out['color_fine'].detach().cpu().numpy()[..., 0, :]
                cv.imwrite(os.path.join(base_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.exr'.format(self.iter_step, i, idx_name)),
                           img_fine)
            else:
                img_fine = render_out['color_fine'].detach().cpu().numpy()
                if self.dataset.convert_linear:
                    img_fine = linear_to_srgb(img_fine)
                img_fine = (img_fine[:, :, 0] * 256).clip(0, 255)
                if self.test_mode:
                    img_out = img_fine
                else:
                    img_out = np.concatenate([img_fine, self.dataset.image_at(idx, resolution_level=resolution_level)])
                cv.imwrite(os.path.join(base_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                           img_out)
            if save_view:
                cv.imwrite(os.path.join(base_dir,
                                        'albedo',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                           albedo_img)
            cv.imwrite(os.path.join(base_dir,
                                    'specular',
                                    '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                       specular_img)
        if 'edge_mask' in render_out:
            edge_mask = render_out['edge_mask'].detach().cpu().numpy().astype(np.uint8) * 255
            os.makedirs(os.path.join(base_dir, 'edge_mask'), exist_ok=True)
            if save_view:
                cv.imwrite(os.path.join(base_dir,
                                        'edge_mask',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx_name)),
                           edge_mask)

    def validate_mesh(self, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


if __name__ == '__main__':
    print('Hello Warden')  # see through smoke

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--data_sub', type=int, default=None)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_scene', action='store_true')
    parser.add_argument('--test_mode', action='store_true')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, **vars(args))

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('validate_image'):
        arg_list = args.mode.split('_')
        if len(arg_list) >= 3:
            _, _, img_idx = arg_list
            img_idx = int(img_idx)
            runner.validate_image(idx=img_idx)
        else:
            resolution_level = 2
            for img_idx in tqdm(range(runner.dataset.n_images)):
                runner.validate_image(idx=img_idx, resolution_level=resolution_level, save_view=img_idx == 0)
    elif args.mode.startswith('validate_view'):
        runner.color_weight = 1.0
        arg_list = args.mode.split('_')
        if len(arg_list) >= 3:
            img_idx = int(arg_list[2])
        else:
            img_idx = 0
        use_phong = True
        if len(arg_list) >= 4:
            assert arg_list[3] in ['phong', 'shading']
            use_phong = arg_list[3] == 'phong'
        if use_phong:
            runner.renderer.mock_phong = True
            runner.renderer.ambient_coeff = 3e-2
            save_name = 'novel_view'
            render_visibility = False
            runner.renderer.use_point_light = False
            runner.renderer.light_intensity = 1.0
        else:
            save_name = f'novel_view_shading_{img_idx}'
            render_visibility = True
        runner.iron_tracer.mock_floor = False
        n_frames = 30 * 2 * 3
        resolution_level = 2 if isinstance(runner.dataset, real_dataset.RealDataset) else 1
        camera = runner.dataset.get_camera(img_idx, resolution_level)
        euler_angle, radius, center = pose_to_euler_radius(camera.C2W.cpu().numpy())
        _, radius_light, _ = pose_to_euler_radius(runner.dataset.pose_sun_all[img_idx].cpu().numpy())
        cur_phi_dir = 1
        cur_theta_dir = 1
        phi_min, phi_max = 5, 45
        phi_min, phi_max = -90 - phi_max, -90 - phi_min
        phi_min = min(phi_min, euler_angle[0])
        phi_max = max(phi_max, euler_angle[0])
        theta_step = 360 * 3 / n_frames
        phi_step = 2 * (phi_max - phi_min) / n_frames
        for i in tqdm(range(n_frames)):
            camera.C2W = torch.from_numpy(euler_radius_to_pose(euler_angle, radius, center)).float().to(runner.device)
            camera.W2C = torch.inverse(camera.C2W)
            if use_phong:
                light_pose = torch.from_numpy(
                    euler_radius_to_pose(euler_angle, radius_light, center)).float().unsqueeze(
                    0).to(runner.device)
            else:
                light_pose = None
            if len(glob(f'{runner.base_exp_dir}/{save_name}/validations_fine/*_0_{i:03d}.png')) > 0:
                print(f'Skipping {i}')
            else:
                runner.validate_image(img_idx, resolution_level, base_dir=f'{runner.base_exp_dir}/{save_name}',
                                      camera=camera, idx_name=f'{i:03d}', render_visibility=render_visibility,
                                      light_pose=light_pose)
            euler_angle[0] += phi_step * cur_phi_dir
            euler_angle[2] += theta_step * cur_theta_dir
            if euler_angle[0] >= phi_max:
                euler_angle[0] = phi_max
                cur_phi_dir *= -1
            elif euler_angle[0] <= phi_min:
                euler_angle[0] = phi_min
                cur_phi_dir *= -1
    elif args.mode.startswith('validate_relight'):
        s_val = min(1e-3, 1 / np.exp(10 * runner.deviation_network.variance.item()))
        runner.renderer.n_samples *= 2
        arg_list = args.mode.split('_')
        if len(arg_list) >= 3:
            img_idx = int(arg_list[2])
        else:
            img_idx = 0
        if len(arg_list) >= 4:
            assert arg_list[3] in ['point', 'dir']
            runner.renderer.use_point_light = arg_list[3] == 'point'
        else:
            runner.renderer.use_point_light = True
        if len(arg_list) >= 5:
            assert arg_list[4] in ['gold', 'emerald']
            save_name = f'novel_light_{arg_list[4]}'
            runner.renderer.mock_phong = True
            runner.renderer.floor_phong = False
            if arg_list[4] == 'gold':
                runner.renderer.phong_albedo = torch.tensor(srgb_to_linear(np.array([0., 0.84313725, 1.]))).float()
                runner.renderer.phong_specular[:] = 0
                runner.renderer.phong_specular.view(3, 9)[:, 2] = torch.tensor(
                    srgb_to_linear(np.array([0., 0.84313725, 1.]))).float()
            elif arg_list[4] == 'emerald':
                runner.renderer.phong_albedo = torch.tensor(
                    srgb_to_linear(np.array([0.46666667, 0.60784314, 0.]))).float()
                runner.renderer.phong_specular[:] = 0
                runner.renderer.phong_specular.view(3, 9)[:, 0] = torch.tensor(
                    srgb_to_linear(np.array([1., 1., 1.]))).float()
        else:
            save_name = 'novel_light'
        n_frames = 30 * 2 * 3
        resolution_level = 2 if isinstance(runner.dataset, real_dataset.RealDataset) else 1
        camera = runner.dataset.get_camera(img_idx, resolution_level)
        euler_angle_cam, _, _ = pose_to_euler_radius(camera.C2W.cpu().numpy())
        euler_angle, radius, center = pose_to_euler_radius(runner.dataset.pose_sun_all[img_idx].cpu().numpy())
        cur_phi_dir = 1
        cur_theta_dir = 1
        if runner.renderer.use_point_light:
            phi_min, phi_max = 15, 45
        else:
            phi_min, phi_max = 0, 30
        phi_min, phi_max = -90 - phi_max, -90 - phi_min
        euler_angle[0] = phi_min
        euler_angle[2] = euler_angle_cam[2]
        theta_step = 360 * 3 / n_frames
        phi_step = 2 * (phi_max - phi_min) / n_frames
        for i in tqdm(range(n_frames)):
            light_pose = torch.from_numpy(euler_radius_to_pose(euler_angle, radius, center)).float().unsqueeze(0).to(
                runner.device)
            if len(glob(f'{runner.base_exp_dir}/{save_name}/validations_fine/*_0_{i:03d}.png')) > 0:
                print(f'Skipping {i}')
            else:
                runner.validate_image(img_idx, resolution_level, base_dir=f'{runner.base_exp_dir}/{save_name}',
                                      idx_name=f'{i:03d}', light_pose=light_pose, save_view=i == 0, s_val=s_val)
            euler_angle[0] += phi_step * cur_phi_dir
            euler_angle[2] += theta_step * cur_theta_dir
            if euler_angle[0] >= phi_max:
                euler_angle[0] = phi_max
                cur_phi_dir *= -1
            elif euler_angle[0] <= phi_min:
                euler_angle[0] = phi_min
                cur_phi_dir *= -1
    elif args.mode.startswith('validate_env'):
        s_val = min(1e-3, 1 / np.exp(10 * runner.deviation_network.variance.item()))
        runner.renderer.n_samples *= 2
        arg_list = args.mode.split('_')
        if len(arg_list) >= 3:
            img_idx = int(arg_list[2])
        else:
            img_idx = 0
        if len(arg_list) >= 4:
            phase = float(arg_list[3])
            save_name = f'env_{phase:.2f}'
        else:
            phase = 0.5
            save_name = 'env'
        runner.renderer.use_point_light = False
        runner.iron_tracer.mock_floor = False
        theta_res = 32
        phi_res = 16
        resolution_level = 2 if isinstance(runner.dataset, real_dataset.RealDataset) else 1
        euler_angle, radius, center = pose_to_euler_radius(runner.dataset.pose_sun_all[img_idx].cpu().numpy())
        for i in range(theta_res):
            euler_angle[2] = (i + phase) / theta_res * 360
            for j in range(phi_res):
                euler_angle[0] = (j + 0.5) / phi_res * 180
                light_pose = torch.from_numpy(euler_radius_to_pose(euler_angle, radius, center)).float().unsqueeze(
                    0).to(runner.device)
                runner.validate_image(img_idx, resolution_level, base_dir=f'{runner.base_exp_dir}/{save_name}',
                                      idx_name=f'theta_{i}_phi_{j}', light_pose=light_pose,
                                      save_view=(i == 0 and j == 0), save_exr=True, s_val=s_val)
    elif args.mode.startswith('validate_normal_depth'):
        img_idx = 0
        runner.validate_normal_depth(idx=img_idx)
