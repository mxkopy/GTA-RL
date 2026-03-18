import cv2
from environment import VideoState
import config

VideoState.init_cuda_arrays()

print("Showing")
while True:
    keypress = cv2.waitKey(1)
    # velocity, depth, distances = VideoState.pop_velocity_and_depth()
    # velocity = velocity.permute(1, 2, 0)
    # cv2.imshow("Distances", distances.squeeze().cpu().numpy() / distances.max().item())
    depth = VideoState.pop_depth()
    depth = VideoState.linearize_depth(depth)
    voxels = VideoState.voxelize(depth).squeeze().sum(dim=0)
    # print(voxels.shape)
    cv2.imshow("Voxels", voxels.cpu().numpy() / config.voxel_depth )
    cv2.imshow("Depth", depth.squeeze().cpu().numpy()) #/ depth.max().item()) 
    # cv2.imshow("Velocity", velocity.squeeze().cpu().numpy() / velocity.abs().max().item() )
    # cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())