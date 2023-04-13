import json
from glob import glob

import cv2
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from models.base_dataset import srgb_to_linear, process_vis, BaseDataset


class NeRFDataset(BaseDataset):
    def __init__(self, conf):
        super().__init__(conf)
        print('Load data: Begin')
        self.data_dir = conf.get_string('data_dir')
        self.split = conf.get_string('split')
        self.convert_linear = conf.get_bool('convert_linear')
        self.load_color = conf.get_bool('load_color', True)
        self.load_mask = conf.get_bool('load_mask', True)
        self.load_vis = conf.get_bool('load_vis', False)
        self.data_sub = conf.get_int('data_sub', None)
        self.test_one_mask = conf.get_bool('test_one_mask', False)
        self.n_resolution_level = conf.get_int('n_resolution_level', 1)
        self.load_sun = conf.get_bool('load_sun')
        self.dilate = conf.get_bool('dilate', False)

        # load/parse metadata
        meta_fname = "{}/transforms_{}.json".format(self.data_dir, self.split)
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if self.data_sub is not None and self.data_sub != -1:
            self.list = self.list[:self.data_sub]

        if self.load_sun:
            self.pose_sun_all = [
                torch.from_numpy(self.parse_raw_camera(np.array(frame['transform_matrix_sun']))).float()
                for frame in self.list]
            self.pose_sun_all = torch.stack(self.pose_sun_all).to(self.device)  # [n_images, 4, 4]

        self.images_lis = [f"{self.data_dir}/{x['file_path']}.png" for x in self.list]
        self.n_images = len(self.images_lis)
        images_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.images_lis])
        images_np = images_np.astype(float) / (np.iinfo(images_np.dtype).max + 1)
        masks_np = images_np[..., 3:].repeat(3, axis=-1)
        images_np = images_np[..., :3]
        images_np = images_np * masks_np + 1 * (1 - masks_np)  # white background
        if self.test_one_mask:  # mock masks with the one in test_one
            image_one = cv.imread(f'{self.data_dir}/test_one/r_000.png', cv.IMREAD_UNCHANGED)
            image_one = image_one.astype(float) / (np.iinfo(image_one.dtype).max + 1)
            mask_one = image_one[..., 3:].repeat(3, axis=-1)
        else:
            mask_one = masks_np[0]
        if self.dilate:
            mask_one = self.dilate_mask(mask_one)
        masks_np = mask_one[None, ...].repeat(masks_np.shape[0], axis=0)

        if self.convert_linear:
            images_np = srgb_to_linear(images_np)
        self.H, self.W = images_np.shape[1], images_np.shape[2]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.meta["camera_angle_x"])

        self.intrinsics_all = [torch.tensor([
            [self.focal, 0, self.W / 2, 0],
            [0, self.focal, self.H / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32) for _ in range(self.n_images)]
        self.pose_all = [torch.from_numpy(self.parse_raw_camera(np.array(frame['transform_matrix']))).float()
                         for frame in self.list]

        if self.load_vis:
            vis_all = []
            for i in range(self.n_images):
                file_path = f"{self.data_dir}/{self.list[i]['file_path']}_white_.png"
                vis = cv.imread(file_path, cv.IMREAD_UNCHANGED)
                vis = process_vis(vis)
                vis = torch.from_numpy(vis).float()
                vis_all.append(vis)
            self.vis_all = torch.stack(vis_all).to(self.device)
            self.vis_all = self.process_resolution(self.vis_all)

        self.images = torch.from_numpy(images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(masks_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.images = self.process_resolution(self.images)
        self.masks = self.process_resolution(self.masks)

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.pose_all_inv = torch.inverse(self.pose_all)
        self.image_pixels = self.H * self.W

        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = np.array(conf['object_bbox_min'])
        self.object_bbox_max = np.array(conf['object_bbox_max'])

        def parse_floor_height(file_path):
            obj = json.load(open(file_path))
            transform_ground = obj['frames'][0]['transform_ground']
            floor_height = transform_ground[2][3]
            return floor_height

        self.floor_height = parse_floor_height(f'{self.data_dir}/transforms_floor.json')

        self.all_pixels = self.get_all_pixels()

        depth_gt = cv2.imread(glob(f'{self.data_dir}/normal/*_depth_*.exr')[0], cv2.IMREAD_UNCHANGED)[..., 0]
        self.depth_gt = torch.from_numpy(depth_gt).float().to(self.device)
        normal_gt = cv2.imread(glob(f'{self.data_dir}/normal/*_normal_*.exr')[0], cv2.IMREAD_UNCHANGED)[..., ::-1]
        normal_gt = np.einsum('ij,NMj->NMi', self.pose_all[0, :3, :3].cpu().numpy().T,
                              normal_gt)  # normal transformation matrix
        normal_gt[..., [1, 2]] = -normal_gt[..., [1, 2]]
        self.normal_gt = torch.from_numpy(normal_gt).float().to(self.device)
        self.normal_gt = F.normalize(self.normal_gt, dim=-1)

        print('Load data: End')

    @staticmethod
    def parse_raw_camera(pose_raw):
        """
        Convert blendshape coordinate space to OpenGL space
        pose_raw: [4, 4]
        return: [4, 4] c2w pose
        """
        pose_flip = np.diag([1, -1, -1, 1])
        pose = pose_raw @ pose_flip
        # pose = np.linalg.inv(pose)
        return pose
