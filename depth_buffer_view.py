import cv2
import torch
from environment import VideoState
from ipc import Channel
from struct import pack

p = 100
pscale = Channel(4, "scale")

from graphics_test import RayCasts, DebugArrows, ShaderVars
import time
from graphics_test import RAY_AT_NDC

while True:
    keypress = cv2.waitKey(10)
    if keypress == 119: # W
        p += 1
        print(p/100)
    if keypress == 115: # S
        p -= 1
        print(p/100)
    pscale.push_nbl(pack('@f', p / 100))
    velocity, depth = VideoState.pop_depth()
    # velocity = (velocity / velocity.square().sum(dim=0).sqrt().max()).permute(1, 2, 0)
    velocity = torch.square(velocity).sum(dim=0).sqrt()
    velocity = velocity / velocity.max()
    # depth = VideoState.linearize_depth(depth)
    rgb = VideoState.pop_rgb()
    cv2.imshow("Depth", (depth / depth.max()).squeeze().cpu().numpy())
    cv2.imshow("Velocity", velocity.squeeze().cpu().numpy())
    cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())