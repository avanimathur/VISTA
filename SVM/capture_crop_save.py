# capture_crop_save.py
import cv2
import numpy as np
import os
from utils.image_utils import extract_center_crop, rgb_normalize, flatten_features

# Get label
label = input("Enter vegetable name :").strip().lower()
folder = "svm_dataset"
os.makedirs(folder, exist_ok=True)

data = []
labels = []

# Open camera
cap = cv2.VideoCapture(0)
print("📸 Press 's' to save (only circular center), 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clone original for preview
    preview = frame.copy()

    # Draw visual circular ROI on screen for guidance
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1]) // 2
    cv2.circle(preview, center, radius, (0, 255, 0), 2)

    # Display full frame with circle guide
    cv2.imshow("Full Camera View", preview)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Extract + normalize only the circular center part
        crop = extract_center_crop(frame, output_size=radius * 2)
        norm = rgb_normalize(crop)
        vec = flatten_features(norm)

        data.append(vec)
        labels.append(label)
        print(f"✅ Captured circular crop for: {label}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to file
np.savez(f"{folder}/{label}_data.npz", data=data, labels=labels)
print(f"📁 Saved to {folder}/{label}_data.npz")
