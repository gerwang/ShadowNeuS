import json
import os

import cv2
import kornia
import numpy as np
import torch

from models.base_dataset import BaseDataset, srgb_to_linear

try:
    import OpenEXR as exr
    import Imath

    openexr = True
except:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    openexr = False


def readEXR(filename):
    """Read color + depth data from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.

    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """

    exrfile = exr.InputFile(filename)
    header = exrfile.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.frombuffer(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C
    Z = channelData['R']
    return Z


def get_scale_mat(points, target_radius):
    min_c, max_c = points.min(dim=0).values, points.max(dim=0).values
    center = (min_c + max_c) / 2
    # move center to 0
    bound = max_c - min_c
    bound_max = bound.max()
    # scale bound_max to target_radius * 2
    trans = -center
    scale = target_radius * 2 / bound_max
    scale_mat = torch.vstack([
        torch.hstack([torch.eye(3) * scale, trans[:, None] * scale]),
        torch.tensor([[0, 0, 0, 1.0]]),
    ])
    return scale_mat


def depth_map_to_pointcloud(depth_map, K, RT, w, h):
    xyz: torch.Tensor = kornia.geometry.depth_to_3d(depth_map.reshape(1, 1, w, h), K, normalize_points=False).squeeze()

    xyz = xyz.reshape(3, -1).permute(1, 0)
    xyz = kornia.geometry.convert_points_to_homogeneous(xyz)
    xyz = (RT.inverse().squeeze() @ xyz.T).T
    xyz = kornia.geometry.convert_points_from_homogeneous(xyz)  # .permute(1, 0)
    xyz = xyz.reshape(w, h, 3)
    return xyz


class DeepShadowDataset(BaseDataset):
    def __init__(self, conf):
        super().__init__(conf)
        print('Load data: Begin')
        self.root = conf.get_string('data_dir')
        self.convert_linear = conf.get_bool('convert_linear')
        self.load_color = conf.get_bool('load_color', True)
        self.load_mask = conf.get_bool('load_mask', True)
        self.load_vis = conf.get_bool('load_vis', False)
        self.data_sub = conf.get_int('data_sub', None)
        self.dilate = conf.get_bool('dilate', False)
        self.n_resolution_level = conf.get_int('n_resolution_level', 1)

        self.normalize = conf.get_bool('normalize', True)
        self.target_radius = conf.get_float('target_radius', 0.99)

        self.light_pos = {}
        lines = self._read_list(os.path.join(self.root, 'all_object_lights.txt'))
        if self.data_sub is not None and self.data_sub != -1:
            lines = lines[:self.data_sub]
        for line in lines:
            name, x, y, z = line.split()
            self.light_pos[name] = np.asarray((x, y, z), dtype=float)

        self.n_images = len(self.light_pos)

        json_file = open(os.path.join(self.root, 'params.json'))
        self.params = json.load(json_file)
        json_file.close()

        self.focal = self.params['focal_length']
        dummy_img_path = os.path.join(self.root, "0", [x for x in self.light_pos.keys()][0] + "_img.png")
        dummy_img = cv2.imread(dummy_img_path)
        self.H, self.W, self.C = dummy_img.shape  # original size

        path_base = dummy_img_path.split("_0_0")[0]
        if openexr:
            self.depth_exr = readEXR(path_base + "_depth.exr") if os.path.exists(path_base + "_depth.exr") else None
        else:
            self.depth_exr = cv2.imread(path_base + "_depth.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        self.depth_exr = np.clip(self.depth_exr, a_min=0, a_max=500.)
        self.depth_floor = np.full_like(self.depth_exr, fill_value=self.depth_exr.max())

        self.intrinsics_all = [torch.tensor([
            [self.focal, 0, self.W / 2, 0],
            [0, self.focal, self.H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32) for _ in range(self.n_images)]
        self.pose_all = [torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 1],
        ], dtype=torch.float32) for _ in range(self.n_images)]

        if self.normalize:
            points = depth_map_to_pointcloud(torch.from_numpy(self.depth_exr).to(self.device),
                                             self.intrinsics_all[0][None, :3, :3], self.pose_all[0][None, ...], self.W,
                                             self.H)

            scale_mat = get_scale_mat(points.reshape(-1, 3), self.target_radius)

            # Apply scale_mat to everything needs it
            def transform_point(scale_mat, p):
                p = torch.from_numpy(p).float().to(scale_mat.device)[None, ...]
                p = kornia.geometry.convert_points_to_homogeneous(p)
                p = (scale_mat[None, ...] @ p[..., None])[..., 0]
                p = kornia.geometry.convert_points_from_homogeneous(p)
                p = p[0].detach().cpu().numpy()
                return p

            self.light_pos = {k: transform_point(scale_mat, v) for k, v in self.light_pos.items()}
            self.depth_exr *= scale_mat[0, 0].item()
            self.depth_floor *= scale_mat[0, 0].item()
            self.pose_all = [torch.tensor([
                [1, 0, 0, scale_mat[0, 3].item()],
                [0, -1, 0, scale_mat[1, 3].item()],
                [0, 0, -1, 1 * scale_mat[2, 2].item() + scale_mat[2, 3].item()],
                [0, 0, 0, 1],
            ], dtype=torch.float32) for _ in range(self.n_images)]

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.pose_all_inv = torch.inverse(self.pose_all)

        self.floor_height = depth_map_to_pointcloud(torch.from_numpy(self.depth_floor).to(self.device),
                                                    self.intrinsics_all[0][None, :3, :3], self.pose_all[0][None, ...],
                                                    self.W, self.H)[0, 0, 2].item()

        normal_gt_path = path_base + "_normal.png"
        self.normal_gt = (cv2.cvtColor(cv2.imread(normal_gt_path), cv2.COLOR_BGR2RGB).astype(
            np.float32) / 255.0) * 2 - 1
        self.silhouette_gt = cv2.imread(path_base + "_silhouette.png")[:, :, 0] / 255.0
        if self.dilate:
            self.silhouette_gt = self.dilate_mask(self.silhouette_gt)

        depth_img = self.depth_exr
        depth_img = depth_img - depth_img.min()
        depth_img /= depth_img.max()
        self.depth_exr_normed = depth_img

        self.normal_gt = torch.from_numpy(self.normal_gt).float().to(self.device)
        self.depth_gt = torch.from_numpy(self.depth_exr).float().to(self.device)
        self.silhouette_gt = torch.from_numpy(self.silhouette_gt).float().to(self.device)

        self.pose_sun_all = torch.stack([torch.vstack([
            torch.hstack([torch.eye(3), torch.from_numpy(pos).float().to(self.device)[:, None]]),
            torch.tensor([[0, 0, 0, 1.0]])]) for pos in self.light_pos.values()])

        self.img_names = [x for x in self.light_pos.keys()]

        self.images = []
        self.vis_all = []

        for i in range(self.n_images):
            img_name = self.img_names[i]
            img_fname = os.path.join(self.root, "0", img_name + "_shadow1.png")
            shadow = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            img_fname = os.path.join(self.root, "0", img_name + "_img.png")
            img = cv2.imread(img_fname).astype(np.float32) / 255.0
            if self.convert_linear:
                img = srgb_to_linear(img)

            self.vis_all.append(torch.from_numpy(shadow).float())
            self.images.append(torch.from_numpy(img).float())

        self.vis_all = torch.stack(self.vis_all).to(self.device)
        self.images = torch.stack(self.images).to(self.device)
        self.masks = self.silhouette_gt[None, ..., None].repeat(self.n_images, 1, 1,
                                                                self.C).to(self.device)  # [n_images, H, W, 3]
        self.vis_all = self.process_resolution(self.vis_all)
        self.images = self.process_resolution(self.images)
        self.masks = self.process_resolution(self.masks)

        self.all_pixels = self.get_all_pixels()
        print('Load data: End')

    @staticmethod
    def _read_list(list_path):
        with open(list_path) as f:
            lists = f.read().splitlines()
        return lists
