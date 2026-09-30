# ============================================================
#  train.py  —  Digit Recognition Model (Machine Learning)
#  Part 1 : Importing libraries
#
#  This file teaches the computer to recognize handwritten
#  digits (0-9), measures its accuracy, and saves the model.
#
#  Project type : Supervised Learning -> Multi-class Classification
#  Model        : SVM (Support Vector Machine)
#  Vision tool  : OpenCV
# ============================================================

# ---- Libraries (ready-made tools we pull in from the top) ----
import cv2                        # OpenCV - image and camera work
import joblib                     # save / load the trained model to a file
import matplotlib                 # library for making graphs / charts
matplotlib.use("Agg")            # "Agg" = build image files without a window
import matplotlib.pyplot as plt   # drawing tool - now "plt" makes charts
import numpy as np                # NumPy - image = grid of numbers, fast handling
from sklearn import datasets, metrics, svm
#     datasets -> ready-made digit images
#     metrics  -> tools to measure accuracy
#     svm      -> our model (Support Vector Machine)
from sklearn.model_selection import train_test_split
#     tool to split data into training and test sets


# ---- Settings ----
IMG_SIZE = 8   # each image is 8x8 pixels (small = fast to train)


def load_data():
    """Load the built-in handwritten digits dataset (1797 images)."""

    digits = datasets.load_digits()        # get the ready-made images + answers

    # Each image is an 8x8 grid. Flatten it into one row of 64 numbers,
    # because the SVM model wants a flat list, not a 2D grid.
    X = digits.images.reshape(len(digits.images), -1)   # the images (inputs)
    y = digits.target                                    # the answers (0-9)

    labels = [str(i) for i in range(10)]   # names: "0","1",...,"9"

    return X, y, labels


def train_model(X, y):
    """Split data, train the SVM, return the model and test data."""

    # Split into 75% training and 25% testing.
    # The model learns from the training part, and we check it
    # on the test part - data it has NEVER seen. That makes the
    # accuracy honest (no cheating).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create the SVM model.
    clf = svm.SVC(gamma=0.001, C=1.0)

    # Train it - this is where the learning actually happens.
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def evaluate(clf, X_test, y_test, labels):
    """Test the model on unseen data and save an accuracy chart."""

    # Ask the model to predict the test images.
    y_pred = clf.predict(X_test)

    # Compare predictions with the real answers -> accuracy.
    acc = metrics.accuracy_score(y_test, y_pred)

    # Print the accuracy nicely - THIS is your screenshot number.
    print("\n" + "=" * 45)
    print(f"  TEST ACCURACY : {acc * 100:.2f}%")
    print("=" * 45 + "\n")

    # Detailed report (per-digit precision/recall).
    print(metrics.classification_report(y_test, y_pred, target_names=labels))

    # Build a confusion matrix - a grid showing which digit
    # got confused with which. Great for the screenshot too.
    cm = metrics.confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - Accuracy {acc * 100:.2f}%")

    # Write the count inside each cell.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)   # save the image file
    print("Saved -> confusion_matrix.png")

    return acc


def main():
    """Run the whole pipeline: load -> train -> evaluate -> save."""

    print("Loading data...")
    X, y, labels = load_data()                 # Part 2
    print(f"Total images : {len(X)}")

    print("Training model...")
    clf, X_test, y_test = train_model(X, y)    # Part 3

    evaluate(clf, X_test, y_test, labels)      # Part 4

    # Save the trained model so webcam.py can use it later.
    joblib.dump({"model": clf, "labels": labels}, "model.pkl")
    print("Saved -> model.pkl")


# This runs main() only when you start the file directly.
if __name__ == "__main__":
    main()