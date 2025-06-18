# train_svm.py
import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and combine all data
data, labels = [], []

for file in glob.glob("svm_dataset/*.npz"):
    npz = np.load(file, allow_pickle=True)
    data.extend(npz["data"])
    labels.extend(npz["labels"])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(data)

# Train SVM
print("🔄 Training SVM model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Predict on training data
y_pred = clf.predict(X)

# Print evaluation metrics
acc = accuracy_score(y, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))

# Save model and encoder
joblib.dump(clf, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n💾 Model and encoder saved.")
