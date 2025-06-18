# utils/image_utils.py
import cv2
import numpy as np

def extract_center_crop(img, output_size=64):
    h, w = img.shape[:2]
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], output_size//2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    cropped = cv2.bitwise_and(img, img, mask=mask)
    x1, y1 = center[0]-radius, center[1]-radius
    x2, y2 = center[0]+radius, center[1]+radius
    return cropped[y1:y2, x1:x2]

def rgb_normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def flatten_features(img, size=(32, 32)):
    img = cv2.resize(img, size)
    return img.flatten() / 255.0  # normalized vector
