import cv2 as cv
import kornia
import numpy as np
import torch
from scipy.spatial.transform.rotation import Rotation as R

from models.raytracer import Camera


# adapted from https://github.com/cgtuebingen/Neural-PIL/blob/23e4aa137895dc8ee02dcc2018cfc49490dd93ad/nn_utils/math_utils.py
def saturate(x, low=0.0, high=1.0):
    return np.clip(x, low, high)


def srgb_to_linear(x):
    x = saturate(x)

    switch_val = 0.04045
    return np.where(
        np.greater_equal(x, switch_val),
        np.power((np.maximum(x, switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92,
    )


def linear_to_srgb(x):
    x = saturate(x)

    switch_val = 0.0031308
    return np.where(
        np.greater_equal(x, switch_val),
        1.055 * np.power(np.maximum(x, switch_val), 1.0 / 2.4) - 0.055,
        x * 12.92,
    )


def process_vis(vis):
    if vis.ndim == 3:  # BGR
        vis = cv.cvtColor(vis, cv.COLOR_BGRA2GRAY)
    vis = vis.astype(np.float32) / (np.iinfo(vis.dtype).max + 1)
    # binarize vis
    threshold = 0.5
    vis[vis > threshold] = 1.0
    vis[vis <= threshold] = 0.0
    return vis


def pose_to_euler_radius(pose):
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    direction = rot @ np.array([0, 0, 1])
    t = -trans[2] / direction[2]
    center = trans + t * direction
    euler_angle = R.from_matrix(rot).as_euler('xyz', degrees=True)
    radius = np.linalg.norm(trans - center)
    return euler_angle, radius, center


def euler_radius_to_pose(euler_angle, radius, center):
    rot = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()
    trans = rot @ np.array([0, 0, -radius]) + center
    ret = np.eye(4)
    ret[:3, :3] = rot
    ret[:3, 3] = trans
    return ret


class BaseDataset:
    def __init__(self, conf):
        self.device = torch.device('cuda')
        self.conf = conf
        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = np.array(conf['object_bbox_min'])
        self.object_bbox_max = np.array(conf['object_bbox_max'])

        self.pose_sun_all = None
        self.n_images = 0
        self.H, self.W = 0, 0
        self.n_resolution_level = 1
        self.convert_linear = False
        self.load_mask = False
        self.load_color = False
        self.load_vis = False
        self.load_normal = False

        self.vis_all = None
        self.images = None
        self.masks = None

        self.intrinsics_all = None
        self.intrinsics_all_inv = None
        self.pose_all = None
        self.pose_all_inv = None

        self.floor_height = 0
        self.all_pixels = None

        self.normal_gt = None
        self.depth_gt = None

    @staticmethod
    def dilate_mask(mask):
        dilatation_size = mask.shape[0] // 40
        dilation_shape = cv.MORPH_ELLIPSE
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                           (dilatation_size, dilatation_size))
        mask = cv.dilate(mask, element)
        return mask

    def process_resolution(self, x):
        cur_level = 1
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        res_dict = {cur_level: x}
        for _ in range(self.n_resolution_level - 1):
            cur_level *= 2
            x = kornia.geometry.transform.resize(x, (x.shape[-2] // 2, x.shape[-1] // 2), interpolation='bilinear',
                                                 align_corners=False)
            res_dict[cur_level] = x
        for key, value in res_dict.items():
            if value.ndim == 4:
                value = value.permute(0, 2, 3, 1)
            res_dict[key] = value.cpu()
        return res_dict

    def get_all_pixels(self):
        res_dict = {}
        cur_resolution_level = 1
        for _ in range(self.n_resolution_level):
            tx = torch.arange(self.W // cur_resolution_level)
            ty = torch.arange(self.H // cur_resolution_level)
            pixels = torch.stack(torch.meshgrid(tx, ty, indexing='xy'), dim=-1)
            res_dict[cur_resolution_level] = pixels
            cur_resolution_level *= 2
        return res_dict

    @staticmethod
    def get_view_light_idx(img_idx):
        return 0, img_idx

    def random_image_idx_same_view(self, view_idx, num_images, exclude_idx=None):
        if exclude_idx is None:
            select_set = np.arange(self.n_images)
        else:
            select_set = np.concatenate([np.arange(exclude_idx), np.arange(exclude_idx + 1, self.n_images)])
        return np.random.choice(select_set, num_images, replace=False)

    def get_camera(self, image_idx, resolution_level):
        camera = Camera(W=self.W, H=self.H,
                        K=self.intrinsics_all[image_idx],
                        K_inv=self.intrinsics_all_inv[image_idx],
                        W2C=self.pose_all_inv[image_idx],
                        C2W=self.pose_all[image_idx])
        if resolution_level != 1:
            camera, _ = camera.resize(1.0 / resolution_level)
        return camera

    def get_light_center_and_ray(self, img_idx, light_pose=None):
        if isinstance(img_idx, int):
            img_idx = [img_idx]
        if light_pose is None:
            light_pose = self.pose_sun_all[img_idx]
        rays_v = torch.tensor([0, 0, 1], dtype=torch.float32)[None, ...]  # batch_size, 3
        rays_v = torch.matmul(light_pose[:, :3, :3], rays_v[:, :, None])[..., 0]
        rays_o = light_pose[:, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return rays_o, rays_v

    def gen_random_pixels_x_pixels_y(self, batch_size, resolution_level):
        this_all_pixels = self.all_pixels[resolution_level]
        n_pixels = this_all_pixels.shape[0] * this_all_pixels.shape[1]
        idx = torch.randperm(n_pixels)[:batch_size]
        return this_all_pixels.view(-1, 2)[idx]

    def gen_camera_ray_data(self, batch_pixels, img_idx, resolution_level):
        pixels_x, pixels_y = batch_pixels.unbind(dim=-1)
        res_dict = {}
        if self.load_mask:
            mask = self.masks[resolution_level][img_idx][(pixels_y, pixels_x)]  # batch_size, 3
            res_dict.update({
                'mask': (mask[:, :1].cuda() > 0.5).float(),
            })
        return res_dict

    def gen_shadow_ray_data(self, batch_pixels, batch_img_idx, resolution_level):
        pixels_x, pixels_y = batch_pixels.unbind(dim=-1)
        res_dict = {}
        if self.load_color:
            color = self.images[resolution_level][batch_img_idx].permute(1, 2, 0, 3)[
                (pixels_y, pixels_x)]  # batch_size, light_size, 3
            res_dict.update({
                'color': color.cuda(),
            })
        if self.load_vis:
            vis = self.vis_all[resolution_level][batch_img_idx].permute(1, 2, 0)[(pixels_y, pixels_x)]
            res_dict.update({
                'vis': vis[..., None].cuda(),
            })
        return res_dict

    @staticmethod
    def near_far_from_sphere(rays_o, rays_d, radius=2.0):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - radius
        far = mid + radius
        return near, far

    def image_at(self, idx, resolution_level):
        novel_res = resolution_level not in self.images
        img = self.images[1 if novel_res else resolution_level][idx].detach().cpu().numpy()
        if self.convert_linear:
            img = linear_to_srgb(img)
        img = (img * 256).clip(0, 255).astype(np.uint8)
        if novel_res:
            img = cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
        return img.clip(0, 255)

    def visibility_at(self, idx, resolution_level):
        novel_res = resolution_level not in self.vis_all
        vis = self.vis_all[1 if novel_res else resolution_level][idx].detach().cpu().numpy()
        vis = (vis * 255).astype(np.uint8)
        if novel_res:
            vis = cv.resize(vis, (self.W // resolution_level, self.H // resolution_level))
        return vis.clip(0, 255)[..., None]
