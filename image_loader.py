import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

import time as tm
import config_file as config

train_file_names = os.listdir(config.PATHNAME+config.TRAIN_DIR_NAME)
test_file_names = os.listdir(config.PATHNAME+config.TEST_DIR_NAME)

# print(len(train_file_names))
# print(len(test_file_names))
np.save('train_file_names.npy', train_file_names)
np.save('test_file_names.npy', test_file_names)

def loadSingleImage(path):
#     loads image in 'bgr' form and converts to 'rgb' and returns rgb variant
#     print(path)
    img = cv2.imread(path)[:, :, ::-1]
    return img

# Convert image to greyscale and returns it
def grayConversion(image, img_format='bgr'):
    # if in 'rgb' form convert to 'bgr'
    if img_format == 'rgb' : img = image[:,:,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Convert image to lab space and returns it
def LABConversion(image, img_format='bgr'):
    if img_format == 'rgb' : img = image[:,:,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img

def rotateImage(image, rot_angle, add_padding=False, scale=1):
    # rotates image by 'rot_angle' degrees
    if add_padding:
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    rows, cols, depth = image.shape
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2), rot_angle, scale)
    image = cv2.warpAffine(image, matrix, (cols,rows))
        
    return image

def flipImage(image, flip_dim='vertical'):
    # to flip the image along central and vertical dimension
    if flip_dim=='vertical':
        img = cv2.flip(image, 1)
    # to flip the image along central and horizontal dimension
    elif flip_dim=='horizontal':
        img = cv2.flip(image, 0)
    elif flip_dim=='upside-down-mirror':
        img = cv2.flip(cv2.flip(image, 0), 1)
    else:
        print("Wrong dimension : using default dimension -> vertical")
        img = cv2.flip(image, 1)

    return img

def alterImage(image):
    random_num = np.random.randint(0, 10)
    
    if random_num is 0:
        return rotateImage(image, 6)
    elif random_num is 1:
        return rotateImage(image, 12)
    elif random_num is 2:
        return rotateImage(image, -12)
    elif random_num is 3:
        return rotateImage(image, -6)
    elif random_num is 4:
        return rotateImage(flipImage(image), 6)
    elif random_num is 5:
        return rotateImage(flipImage(image), 12)
    elif random_num is 6:
        return rotateImage(flipImage(image), -12)
    elif random_num is 7:
        return rotateImage(flipImage(image), -6)
    elif random_num is 8:
        return flipImage(image)
    else:
        return image
    
class DataBatchGenerator():
    def __init__(self):
        # Setting up train and test set directories
        self.path = config.PATHNAME
        self.train_dir_name = config.TRAIN_DIR_NAME
        self.test_dir_name = config.TEST_DIR_NAME
        self.train_path = self.path + self.train_dir_name
        self.test_path = self.path + self.test_dir_name
        
        # File names
        self.train_file_names = train_file_names
        self.test_file_names = test_file_names
        
        # Setting up train and test parameters
        self.num_images_train = config.NUM_TRAIN_IMAGES
        self.num_images_test = config.NUM_TEST_IMAGES

        # Setting up batch parameters
        self.batch_size = config.BATCH_SIZE
        
        # Setting out labels
        self.labels = config.LABELS_DATASET # It must be one-hot-encoded
        self.label_names = config.LABEL_NAMES
        return
    
    def get_train_data_batch(self, batch_size=None):        
        
        batch_size = self.batch_size if batch_size is None else batch_size
        train_file_names = np.load('train_file_names.npy').tolist()
        random_indices = np.random.randint(0, self.num_images_train, size=[batch_size])
        
        lab_batch = []
        gray_batch = []
        indices = []
        labels_batch = []
        for index in random_indices:
            # Load one single image
            img = loadSingleImage(self.train_path+train_file_names[index])
            img = alterImage(img)
            indices.append(index)
            lab_batch.append(LABConversion(img, img_format='rgb'))
            gray_batch.append(grayConversion(img, img_format='rgb'))
            labels_batch.append(config.LABELS_DATASET[index])
            
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch, dtype=np.uint8)
        }
    
    
    def get_test_data_batch(self, batch_size=None):   
        
        batch_size = self.batch_size if batch_size is None else batch_size
        test_file_names = np.load('test_file_names.npy').tolist()
        random_indices = np.random.randint(0, self.num_images_test, size=[batch_size])
        
        lab_batch = []
        gray_batch = []
        indices = []
        labels_batch = []
        for index in random_indices:
            # Load one single image
            img = loadSingleImage(self.test_path+test_file_names[index])
            img = alterImage(img)
            indices.append(index)
            lab_batch.append(LABConversion(img, img_format='rgb'))
            gray_batch.append(grayConversion(img, img_format='rgb'))
            labels_batch.append(config.LABELS_DATASET[index])
            
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch, dtype=np.uint8)
        }