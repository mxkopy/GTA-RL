import cv2
from environment import VideoState


while True:
    keypress = cv2.waitKey(10)
    depth = VideoState.pop_depth()
    rgb = VideoState.pop_rgb()
    cv2.imshow("Depth", (depth * 255).squeeze().cpu().numpy())
    cv2.imshow("RGB", rgb.permute(1, 2, 0).squeeze().cpu().numpy())