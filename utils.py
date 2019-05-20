from PIL import Image
import numpy as np
def show_img(array):
    formatted = (array * 255 / np.max(array)).astype('uint8')
    img = Image.fromarray(formatted,'RGB')
    img.show()
