import cv2
import torch
from environment import VideoState

while True:
    keypress = cv2.waitKey(1)
    cv2.imshow("DepthBuffer", (VideoState.pop_depth() * 255).squeeze().cpu().numpy())
    cv2.imshow("RGB", (VideoState.pop_rgb() * 255).to(dtype=torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy())