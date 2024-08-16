import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_path = r"Deep Learning Project\keras_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    print("Model loaded successfully")
else:
    print(f"Model file not found at {model_path}")
    exit()

# Load the labels
labels_path = r"Deep Learning Project\labels.txt"
if os.path.exists(labels_path):
    class_names = open(labels_path, "r").readlines()
else:
    print(f"Labels file not found at {labels_path}")
    exit()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = "bact6.jpg"
if not os.path.exists(image_path):
    print(f"Image file not found at {image_path}")
    exit()

# Open and resize the image
image = Image.open(image_path).convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict with the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name)
print("Confidence Score:", confidence_score)