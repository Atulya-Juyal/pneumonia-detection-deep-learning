# cnn_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model():
    """
    Builds and compiles the CNN model.
    """
    model = Sequential()

    # ✅ Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # ✅ Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # ✅ Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # ✅ Flatten the output
    model.add(Flatten())

    # ✅ Fully Connected Layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # ✅ Output Layer
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # ✅ Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
