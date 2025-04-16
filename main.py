from load_data import train_generator, val_generator, test_generator
from cnn_model import build_cnn_model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

# ✅ Model path
model_path = './models/cnn_model.h5'

# ✅ Check if model already exists
if os.path.exists(model_path):
    print("\n✅ Model already trained. Loading the saved model...")
    model = load_model(model_path)
else:
    print("\n🚀 Training the model from scratch...")
    model = build_cnn_model()

    # ✅ Train the model
    EPOCHS = 20
    STEPS_PER_EPOCH = len(train_generator)
    VALIDATION_STEPS = len(val_generator)

    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=VALIDATION_STEPS
    )

    # ✅ Save the trained model
    os.makedirs('./models', exist_ok=True)
    model.save(model_path)
    print(f"\n✅ Model saved at {model_path}")

    # ✅ Plot the training history
    def plot_training_history(history):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # ✅ Plot Accuracy
        axs[0].plot(history.history['accuracy'], label='Train Accuracy')
        axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[0].set_title('Model Accuracy')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()

        # ✅ Plot Loss
        axs[1].plot(history.history['loss'], label='Train Loss')
        axs[1].plot(history.history['val_loss'], label='Validation Loss')
        axs[1].set_title('Model Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    # ✅ Display the training history plots
    plot_training_history(history)
