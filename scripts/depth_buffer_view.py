import cv2
from environment import VideoState

# from graphics_test import RayCasts, DebugArrows, ShaderVars
# import time
# from graphics_test import RAY_AT_NDC

VideoState.init_cuda_arrays()

while True:
    keypress = cv2.waitKey(1)
    velocity, depth = VideoState.pop_velocity_and_depth()
    # velocity = (velocity / velocity.square().sum(dim=0).sqrt().max()).permute(1, 2, 0)
    # velocity = torch.square(velocity).sum(dim=0).sqrt()
    # velocity = velocity / velocity.max()
    # depth = VideoState.linearize_depth(depth)
    rgb = VideoState.pop_rgb()
    cv2.imshow("Depth", (depth / depth.max()).squeeze().cpu().numpy())
    # cv2.imshow("Velocity", velocity.squeeze().cpu().numpy())
    cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())