from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
# print(ds)
# print(ds['train']['image'])

from PIL import Image
import numpy as np

def pil_to_0_1_array(image):
    # Convert image to grayscale (mode 'L')
    image_gray = image.convert('L')
    
    # Resize image to 28x28 pixels (if needed)
    image_resized = image_gray.resize((28, 28))
    
    # Convert PIL image to NumPy array
    image_array = np.array(image_resized)
    
    # Apply thresholding to convert pixel values to 0 or 1
    threshold = 128
    image_array_binary = (image_array > threshold).astype(np.uint8)
    
    return image_array_binary.flatten().tolist()  # Return flattened list of 0s and 1s

# Example usage:
# Assuming img is a PIL image object

test=ds['train']['image'][0]
label=ds['train']['label'][0]
image_data = pil_to_0_1_array(test)
print(image_data)
print(label)



