import cv2 as cv
import numpy as np
import torch

from models.base_dataset import srgb_to_linear, BaseDataset


class RealDataset(BaseDataset):
    def __init__(self, conf):
        super().__init__(conf)
        print('Load data: Begin')
        self.data_dir = conf.get_string('data_dir')
        self.convert_linear = conf.get_bool('convert_linear')
        self.load_color = conf.get_bool('load_color', True)
        self.load_mask = conf.get_bool('load_mask', True)
        self.load_vis = False
        self.data_sub = conf.get_int('data_sub', None)
        self.n_resolution_level = conf.get_int('n_resolution_level', 1)
        self.dilate = conf.get_bool('dilate', False)
        self.split = conf.get_string('split', 'train')

        param_name = f'params_{self.split}.npz' if self.split != 'train' else 'params.npz'
        self.params = np.load(f'{self.data_dir}/{param_name}')
        light_pose = self.params['light_pose']
        if self.data_sub is not None and self.data_sub != -1:
            light_pose = light_pose[:self.data_sub]
        self.n_images = light_pose.shape[0]
        self.pose_sun_all = torch.from_numpy(light_pose).float().to(self.device)

        self.floor_height = float(self.params['floor_height'])

        images_np = np.stack([cv.imread(f'{self.data_dir}/image/{i:03d}.jpg') for i in range(self.n_images)])
        images_np = images_np.astype(float) / (np.iinfo(images_np.dtype).max + 1)
        if self.convert_linear:
            images_np = srgb_to_linear(images_np)
        self.H, self.W = images_np.shape[1], images_np.shape[2]
        self.images = torch.from_numpy(images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.images = self.process_resolution(self.images)

        mask_np = cv.imread(f'{self.data_dir}/mask.png')
        mask_np = mask_np.astype(float) / (np.iinfo(mask_np.dtype).max + 1)
        if self.dilate:
            mask_np = self.dilate_mask(mask_np)
        self.masks = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).repeat(self.n_images, 1, 1, 1).to(
            self.device)
        self.masks = self.process_resolution(self.masks)

        self.H, self.W = images_np.shape[1], images_np.shape[2]

        self.intrinsics_all = torch.from_numpy(self.params['camera_intrinsics']).float().unsqueeze(0).repeat(
            self.n_images, 1, 1).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.pose_all = torch.from_numpy(self.params['camera_pose']).float().unsqueeze(0).repeat(
            self.n_images, 1, 1).to(self.device)  # [n_images, 4, 4]
        self.pose_all_inv = torch.inverse(self.pose_all)

        self.all_pixels = self.get_all_pixels()

        print('Load data: End')
