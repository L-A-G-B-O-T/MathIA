from PIL import Image
import numpy as np

img = Image.open("Image_Processing/MonochromeBit.bmp")
arr = np.array(img)

print(arr)
