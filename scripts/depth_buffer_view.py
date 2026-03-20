import config
import cv2
import torch
from environment import VideoState

VideoState.init_cuda_arrays()

print("Showing")
while True:
    keypress = cv2.waitKey(1)
    rgb = VideoState.pop_rgb()
    depth = VideoState.pop_depth()
    depth = VideoState.linearize_depth(depth)
    voxels = VideoState.voxelize(depth).squeeze() * torch.arange(config.voxel_depth, device='cuda').reshape(-1, 1, 1)
    voxels = voxels.sum(dim=0)
    cv2.imshow("Voxels", voxels.cpu().numpy() / config.voxel_depth )
    cv2.imshow("Depth", depth.squeeze().cpu().numpy())
    cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())