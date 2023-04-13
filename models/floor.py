import cv2
import torch
import torch.nn.functional as F


class FloorHeight:
    def __init__(self, floor_height):
        self.floor_height = floor_height
        self.clip_flag = False

    def get_floor_distance(self, rays_o, rays_d, eps=1e-12):
        """
        input: [..., 3]
        output: [..., 1]
        """
        res = (self.floor_height - rays_o[..., 2:]) / (rays_d[..., 2:] + eps)
        return res


class WallFloorHeight(FloorHeight):
    def __init__(self, floor_height, wall_x):
        super().__init__(floor_height)
        self.wall_x = wall_x

    def get_floor_distance(self, rays_o, rays_d, eps=1e-12):
        """
        input: [..., 3]
        output: [..., 1]
        """
        res_base = super().get_floor_distance(rays_o, rays_d, eps)
        res_self = (self.wall_x - rays_o[..., :1]) / (rays_d[..., :1] + eps)
        res = torch.minimum(res_base, res_self)
        return res


class DepthFloorHeight(FloorHeight):
    def __init__(self, floor_height, camera, depth_path, device):
        super().__init__(floor_height)
        self.device = device
        depth_floor = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[..., 0]
        self.depth_floor = torch.from_numpy(depth_floor).float().to(self.device)
        self.camera = camera

    def get_floor_distance(self, rays_o, rays_d, eps=1e-12):
        """
        input: [..., 3]
        output: [..., 1]
        """
        uv = self.camera.project(rays_o + rays_d)
        uv_normed = uv / torch.tensor([self.camera.W, self.camera.H], dtype=torch.float32) * 2 - 1
        floor_depth = F.grid_sample(
            self.depth_floor[None, None, ...].expand(uv.shape[0], 1, *self.depth_floor.shape),
            uv_normed[..., None, None, :],
            mode='bilinear',  # assume the floor is smooth, with no depth discontinuities
            align_corners=False,
            padding_mode='border',
        )[..., 0, 0]
        _, _, rays_d_norm = self.camera.get_rays(uv)
        floor_distance = floor_depth * rays_d_norm[..., None]
        return floor_distance
