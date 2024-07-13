from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
# print(ds)
# print(ds['train']['image'])

from PIL import Image
import numpy as np

def pil_to_0_1_array(image):

    image_gray = image.convert('L')
    image_resized = image_gray.resize((28, 28))
    image_array = np.array(image_resized)
    threshold = 128
    image_array_binary = (image_array > threshold).astype(np.uint8)
    return image_array_binary.flatten().tolist() 

# Example usage:
# Assuming img is a PIL image object

# print(ds)

# train=ds['train']['image'][0]
# array=pil_to_0_1_array(train)
# print(len(array))

print(ds['train'].shape(0))



# test=ds['train']['image'][0]
# label=ds['train']['label'][0]
# image_data = pil_to_0_1_array(test)
# print(image_data)
# print(label)



