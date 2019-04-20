import os
from PIL import Image
import numpy as np
import utils


DATASET_DIR = './'
DATASET_IMAGES = 'Images'
new_width, new_height = 224, 224
class_size = 120
test_split_number = 20


def _download_dogs(path):
    pwd = os.getcwd()
    os.chdir(path)
    if not os.path.isdir(DATASET_IMAGES):
        if not os.path.isfile('images.tar'):
            os.system('wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
        os.system('tar -xvf images.tar')
        os.system('rm images.tar')
    os.chdir(pwd)
    os.system('rm dogs/Images/n02105855-Shetland_sheepdog/n02105855_2933.jpg')
    print('Images downloaded and unpacked')

def get_dog_types(path):
    _download_dogs(DATASET_DIR)
    dog_types = os.listdir(path)
    return dog_types

def substract_mean(data_arr):
    mean = np.array([12.14, 11.52, 99.71])
    data_arr -=  mean
    return data_arr

def prepare_img_data(data):
    #if not os.path.exists("/home/dogs/"):
    #    _download_dogs(DATASET_DIR)
    #train_img_names, train_labels, test_img_names,test_labels = utils.read_image_names_and_assign_labels(class_size,test_split_number,DATASET_IMAGES)
    train_data = utils.resize_and_crop(new_height,new_width,data)
    train_data_arr =  substract_mean(train_data)
    print("method successful")
    return train_data_arr
