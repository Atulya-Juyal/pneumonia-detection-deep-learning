# evaluate.py
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import test_generator

# ✅ Load the Saved Model
model_path = './models/cnn_model.h5'
model = load_model(model_path)
print(f"\n✅ Loaded model from {model_path}")

# ✅ Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# ✅ Generate Classification Report and Confusion Matrix
# Get the true labels and predictions
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.round(y_pred_probs).flatten().astype(int)

# ✅ Print Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# ✅ Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# ✅ Display Confusion Matrix
plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Pneumonia'])
