import json
import os
from argparse import ArgumentParser
from glob import glob

import cv2
from kornia.geometry.transform import resize

from models.base_dataset import linear_to_srgb, pose_to_euler_radius
from models.nerf_dataset import NeRFDataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--work_path', required=True)
    parser.add_argument('--n_theta', type=int, default=32)
    parser.add_argument('--n_phi', type=int, default=16)
    parser.add_argument('--env_paths', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--save_names', required=True)
    parser.add_argument('--norm_coeff', type=float, default=1e-3)
    parser.add_argument('--bg_ratio', type=float, default=0.15)
    parser.add_argument('--super_sample', action='store_true')
    args = parser.parse_args()
    device_ids = args.device_ids.split(',')
    devices = [torch.device(f'{args.device}:{x}') for x in device_ids]
    case_name = os.path.basename(os.path.dirname(args.work_path.rstrip('/')))
    pose = NeRFDataset.parse_raw_camera(
        np.array(json.load(open(f'public_data/nerf_synthetic/{case_name}/transforms_train.json'))['frames'][0][
                     'transform_matrix']))
    euler_angle, _, _ = pose_to_euler_radius(pose)
    camera_phi = euler_angle[0]
    camera_theta = euler_angle[2]

    area_diffs = []
    light_stage_basis = []
    env_names = ['env_0.00', 'env_0.25', 'env', 'env_0.75']

    phi_i = args.n_phi - 1
    for i in range(args.n_phi):
        phi = (phi_i + 0.5) / args.n_phi * np.pi
        theta_i = int(args.n_theta / 2 - 1)
        for j in range(args.n_theta):
            # theta = (theta_i + 0.5) / args.n_theta * np.pi * 2
            area_diffs.append(abs(np.sin(phi)))
            if args.super_sample:
                img_path = f'{args.work_path}/{env_names[theta_i % 4]}/validations_fine/*_0_theta_{theta_i // 4}_phi_{phi_i}.exr'
            else:
                img_path = f'{args.work_path}/env/validations_fine/*_0_theta_{theta_i}_phi_{phi_i}.exr'
            img = cv2.imread(glob(img_path)[0], cv2.IMREAD_UNCHANGED)
            light_stage_basis.append(torch.from_numpy(img).float())
            theta_i -= 1
            if theta_i < 0:
                theta_i += args.n_theta
        phi_i -= 1

    area_diffs = torch.tensor(area_diffs, dtype=torch.float32)
    assert len(light_stage_basis) % len(devices) == 0
    n_slice = len(light_stage_basis) // len(devices)
    light_stage_batch = []
    for i in range(len(devices)):
        light_stage_batch.append(torch.stack(light_stage_basis[i * n_slice:(i + 1) * n_slice]).to(devices[i]))
    alpha_mask = cv2.imread(glob(f'{args.work_path}/env/alpha/*.png')[0])[..., :1] / 255.0

    save_names = args.save_names.split(',')
    env_paths = args.env_paths.split(',')
    for save_name, env_path in zip(save_names, env_paths):
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)[..., :3]
        env_map = torch.from_numpy(env_map).float().to(devices[0])
        shift_step = env_map.shape[1] // args.n_frames
        camera_row = int(-camera_phi / 180 * env_map.shape[0])
        camera_col = int(env_map.shape[1] / 2)
        os.makedirs(f'{args.work_path}/{save_name}', exist_ok=True)
        for i in range(args.n_frames):
            cur_env_map = torch.roll(env_map, shift_step * i, dims=1)
            # convert cam_space envmap to world_space envmap
            world_env_map = torch.roll(cur_env_map, -int(camera_theta / 360 * env_map.shape[1]), dims=1)
            this_env_map = resize(world_env_map.permute(2, 0, 1), (args.n_phi, args.n_theta),
                                  interpolation='bicubic', align_corners=False, antialias=True).permute(1, 2, 0)
            this_env_map = this_env_map.view(-1, 3)
            res_batch = []
            for j in range(len(devices)):
                res_batch.append(
                    (this_env_map[j * n_slice:(j + 1) * n_slice, None, None, :].to(devices[j]) * light_stage_batch[
                        j]).sum(dim=0).to(devices[0]))
            res = torch.stack(res_batch, dim=0).sum(dim=0)
            # res = (this_env_map[:, None, None, :] * light_stage_basis).sum(dim=0)
            res = res * args.norm_coeff
            res = res.cpu().numpy()
            res = linear_to_srgb(res)

            bg_side = int(cur_env_map.shape[0] * args.bg_ratio)
            background_img = cur_env_map[camera_row - bg_side:camera_row + bg_side,
                             camera_col - bg_side:camera_col + bg_side]
            background_img = resize(background_img.permute(2, 0, 1), (res.shape[0], res.shape[1]),
                                    interpolation='bicubic', align_corners=False, antialias=True).permute(1, 2, 0)
            background_img = linear_to_srgb(background_img.cpu().numpy())

            res = res * alpha_mask + background_img * (1.0 - alpha_mask)

            vis_env_height = int(res.shape[0] * 0.15)
            vis_env_width = vis_env_height * args.n_theta // args.n_phi
            if args.super_sample:
                vis_env_width /= 4
            vis_env_width = int(vis_env_width)
            vis_env_map = resize(cur_env_map.permute(2, 0, 1), (vis_env_height, vis_env_width),
                                 interpolation='bicubic', align_corners=False, antialias=True).permute(1, 2, 0)
            vis_env_map = linear_to_srgb(vis_env_map.cpu().numpy())
            res[:vis_env_height, -vis_env_width:] = vis_env_map

            cv2.imwrite(f'{args.work_path}/{save_name}/{i:03d}.png', res * 255)
