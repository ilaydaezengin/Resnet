import os
import numpy as np
from glob import glob
from PIL import Image,ImageOps

def Compose(*operations):
    def transform_all(img):
        for operation in operations:
            img = operation(img)
        return img
    return transform_all

f = Compose("flip(0.5),Jitter(0.02)")

def get_next_batch(perm, data,labels,idx, batch_size):
    if data.shape[0]-idx < batch_size:
        index_array = perm[idx:]
    else:
        index_array = perm[idx:idx+batch_size]
    databatch = data[index_array]
    labelbatch = labels[index_array]
    return np.asarray(databatch), np.asarray(labelbatch)


def read_image_names_and_assign_labels(class_size,test_split_num,path):
    img_paths = glob(path + '/*')
    test_images, test_labels = [],[]
    images, labels = [],[]
    #dog_types = get_dog_types(img_paths)
    #dog_types = os.listdir(path)
    for idx,dog_type in enumerate(img_paths):
        imgs = glob(dog_type + '/*')
        #img_path = os.path.join(path,dog_type)
        #img_list = os.listdir(dog_type)
        for i,img in enumerate(imgs):
            if i < test_split_num:
                test_images.append(img)
                test_labels.append(idx)
            else:
                images.append(img)
                labels.append(idx)
    y_eye = np.eye(class_size)[labels].astype(np.int32)
    test_y_eye = np.eye(class_size)[test_labels].astype(np.int32)
    return np.asarray(images),np.asarray(y_eye),np.asarray(test_images),np.asarray(test_labels)



def resize_and_crop(new_height,new_width,X):
    img_arr = []
    for image in X:
        img = Image.open(image)
        img_height,img_width = img.size
        ratio = 0
        if img_height < img_width:
            ratio = img_height / img_width
            img_width = int(new_width // ratio)
            img_height = new_height
        else:
            ratio = img_width / img_height
            img_height = int(new_height // ratio)
            img_width = new_width
        img = img.resize((img_height,img_width))
        img = ImageOps.fit(img,(new_height,new_width))
        img_array = np.asarray(img)
        img_arr.append(img_array)
        img.close()
    img_arr = np.array(img_arr)
    return np.reshape(img_arr,(-1,new_height,new_width,3))
