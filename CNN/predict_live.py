import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

model = tf.keras.models.load_model("vegetable_classifier.h5")
engine = pyttsx3.init()

CLASS_NAMES = list(np.load("class_names.npy"))

cap = cv2.VideoCapture(0)
IMG_SIZE = 64

print("Press 'p' to predict, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live", frame)
    key = cv2.waitKey(1)

    if key == ord('p'):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        prediction = model.predict(np.expand_dims(img, axis=0))[0]
        class_idx = np.argmax(prediction)
        vegetable = CLASS_NAMES[class_idx]
        print(f"Prediction: {vegetable}")
        engine.say(f"This is {vegetable}")
        engine.runAndWait()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
