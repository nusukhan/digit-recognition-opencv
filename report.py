# ============================================================
#  report.py  —  Handwritten Digit Recognition (SVM)
#  Trains the model and saves a results image:
#  accuracy_report.png  (confusion matrix + classification report)
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Load data
digits = datasets.load_digits()
X = digits.images.reshape(len(digits.images), -1)
y = digits.target
labels = [str(i) for i in range(10)]

# Split into train (75%) and test (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the SVM
clf = svm.SVC(gamma=0.001, C=1.0)
clf.fit(X_train, y_train)

# Evaluate on unseen test data
y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
report_text = metrics.classification_report(y_test, y_pred, target_names=labels)
cm = metrics.confusion_matrix(y_test, y_pred)

# Build the results image (two panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Handwritten Digit Recognition  |  SVM Model  |  Test Accuracy: {acc*100:.2f}%",
    fontsize=15, fontweight="bold",
)

# Left panel: confusion matrix
ax1.imshow(cm, cmap="Blues")
ax1.set_xticks(range(len(labels)), labels)
ax1.set_yticks(range(len(labels)), labels)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
for i in range(len(labels)):
    for j in range(len(labels)):
        ax1.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

# Right panel: classification report as text
ax2.axis("off")
ax2.set_title("Classification Report")
ax2.text(0.0, 0.98, report_text, family="monospace", fontsize=11,
         va="top", ha="left")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("accuracy_report.png", dpi=150)

print(f"Test accuracy: {acc*100:.2f}%")
print("Saved: accuracy_report.png")