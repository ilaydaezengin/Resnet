import numpy as np
from PIL import Image,ImageEnhance
import tensorflow as tf

class transform:

    def Compose(*operations):
        def transform_all(img):
            for operation in operations:
                img = operation(img)
            return img
        return transform_all

    def flip(prob):
        def f(img):
            x = np.random.rand()
            if x < prob:
                img= img.transpose(Image.FLIP_LEFT_RIGHT)
                return img
            else:
                return img
        return f


    def Jitter(brightness):
        def transform_img(img):
            img = ImageEnhance.Brightness(img).enhance(brightness)
            return img
        return transform_img

    def mean():
        def f(img):
            mean = np.mean(img)
            img -= mean
            return img
        return f


    def crop_():
        def crop_img(img):
            width, height = img.size
            left = np.abs((height - width) / 2)
            top = 0
            right = (height + width) / 2
            if width > height:
                img = img.crop((left,top,right,height))
            else:
                img = img.crop((top,left,width,right))
            return img
        return crop_img


    def resize_(given_size):
        def resize_img(img):
            height, width = img.size
            if np.abs(width - given_size) < np.abs(height - given_size):
                ratio = given_size / width
            else:
                ratio = given_size / height
            new_width = int(np.ceil(width * ratio))
            new_height = int(np.ceil(height * ratio))
            img = img.resize((new_height ,new_width))
            crop_img = crop_()
            #crop_img = tf.image.central_crop(img,)
            img = crop_img(img)
            return img
        return resize_img
