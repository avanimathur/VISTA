import cv2
import os

# Ask user for the vegetable label
label = input("Enter vegetable name (e.g., potato, green_chilli): ")
folder = f"dataset/{label}"
os.makedirs(folder, exist_ok=True)

# Open camera
cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_path = os.path.join(folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
