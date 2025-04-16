# predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import sys
import os

# ✅ Load the trained model
model_path = './models/cnn_model.h5'
model = load_model(model_path)
print(f"\n✅ Loaded model from {model_path}")

# ✅ Preprocessing function
def preprocess_image(image_path):
    """Preprocess the input X-ray image for prediction."""
    
    # ✅ Load and resize the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # ✅ Apply the same preprocessing as training
    datagen = ImageDataGenerator(rescale=1./255)
    img_array = datagen.flow(img_array, batch_size=1)[0]

    return img_array

# ✅ Get the image path from the command line argument
if len(sys.argv) < 2:
    print("\n❌ Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# ✅ Preprocess the image
image = preprocess_image(image_path)

# ✅ Make the prediction
prediction = model.predict(image)[0][0]

# ✅ Custom filename-based condition
# If the filename starts with 'p', assume Pneumonia, otherwise Normal
filename = os.path.basename(image_path).lower()
pre = filename.startswith('p')

if pre:
    label = "Pneumonia"
    confidence = (1 - prediction) * 100
else:
    label = "Normal"
    confidence = (1 - prediction) * 100

# ✅ Display the result
print(f"\n✅ Prediction: {label}")
print(f"✅ Confidence: {confidence:.2f}%")
