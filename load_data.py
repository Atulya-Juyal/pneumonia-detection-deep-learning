# load_data.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ✅ Dataset Paths
train_dir = './data/train'
val_dir = './data/val'
test_dir = './data/test'

# ✅ Image Size and Batch Size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# ✅ Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values
    rotation_range=20,        # Rotate images by 20 degrees
    width_shift_range=0.1,    # Horizontal shift
    height_shift_range=0.1,   # Vertical shift
    shear_range=0.1,          # Shear transformation
    zoom_range=0.2,           # Zoom in/out
    horizontal_flip=True,     # Randomly flip images
    fill_mode='nearest'       # Fill missing pixels
)

# ✅ No augmentation for validation and test sets (only rescale)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load Images from Folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'   # Binary classification: pneumonia vs normal
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Keep order for evaluation
)

# ✅ Visualize Some Images
def plot_images(generator):
    """Display a batch of images with labels."""
    x_batch, y_batch = next(generator)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for i in range(12):
        axes[i].imshow(x_batch[i])
        axes[i].set_title('Pneumonia' if y_batch[i] == 1 else 'Normal')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# ✅ Plot and Verify Augmentation
plot_images(train_generator)
