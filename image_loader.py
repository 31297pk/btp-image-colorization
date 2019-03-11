import cv2
import numpy as np
import matplotlib.pyplot as plt

import time as tm

import config_file as config


def loadSingleImage(path):
#     loads image in 'bgr' form and converts to 'rgb' and returns rgb variant
#     print(path)
    img = cv2.imread(path)[:, :, ::-1]
    return img


# In[4]:


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


# In[5]:


class DataBatchGenerator():
    def __init__(self):
        # Setting up train and test set directories
        self.path = config.PATHNAME
        self.train_dir_name = config.TRAIN_DIR_NAME
        self.test_dir_name = config.TEST_DIR_NAME
        self.train_path = self.path + self.train_dir_name
        self.test_path = self.path + self.test_dir_name
        self.img_format = config.IMG_NAME_FORMAT
        
        # Setting up train and test parameters
        self.num_images_train = config.NUM_TRAIN_IMAGES
        self.num_images_test = config.NUM_TEST_IMAGES

        # Setting up batch parameters
        self.batch_size = config.BATCH_SIZE
        
        # Setting out labels
        self.labels = config.LABELS_DATASET # It must be one-hot-encoded
        self.label_names = config.LABEL_NAMES
        return
    
    def get_train_data_batch(self):        
        random_indices = np.random.randint(1, self.num_images_train+1, size=[self.batch_size])
        lab_batch = []
        gray_batch = []
        indices = []
        labels_batch = []
        for index in random_indices:
            
            # Load one single image
            img = loadSingleImage(self.train_path+self.img_format.format(index))
        
            indices.append(index)
            lab_batch.append(LABConversion(img, img_format='rgb'))
            gray_batch.append(grayConversion(img, img_format='rgb'))
            labels_batch.append(config.LABELS_DATASET[index-1])
        
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch, dtype=np.uint8)
        }
    
    def get_test_data_batch(self):
        # Pick up images from dataset to fill up a batch
        random_indices = np.random.randint(1, self.num_images_test+1, size=[self.batch_size])
        
        # Initializing batches
        lab_batch = []
        gray_batch = []
        labels_batch = []
        indices = []
        for index in random_indices:
            # Load one single image
            img = loadSingleImage(self.test_path+self.img_format.format(index))
            indices.append(index)
            lab_batch.append(LABConversion(img, img_format='rgb'))
            gray_batch.append(grayConversion(img, img_format='rgb'))
            labels_batch.append(config.LABELS_DATASET[index-1])
        
        # Return batch in form of dictionary 
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch)
        }
    
    def get_test_image(self, index=None):
        # Pick up images from dataset to fill up a batch
        if index is None : index = np.random.randint(1, self.num_images_test+1, 1)
        
        # Initializing batches
        lab_batch = []
        gray_batch = []
        labels_batch = []
        indices = []
            
        # Load one single image
        img = loadSingleImage(self.test_path+self.img_format.format(index))
        indices.append(index)
        lab_batch.append(LABConversion(img, img_format='rgb'))
        gray_batch.append(grayConversion(img, img_format='rgb'))
        labels_batch.append(config.LABELS_DATASET[index-1])
        
        # Return batch in form of dictionary 
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch)
        }
    
    def get_train_image(self, index=None):
        # Pick up images from dataset to fill up a batch
        if index is None : index = np.random.randint(1, self.num_images_train+1, 1)
        
        # Initializing batches
        lab_batch = []
        gray_batch = []
        labels_batch = []
        indices = []
            
        # Load one single image
        img = loadSingleImage(self.train_path+self.img_format.format(index))
        indices.append(index)
        lab_batch.append(LABConversion(img, img_format='rgb'))
        gray_batch.append(grayConversion(img, img_format='rgb'))
        labels_batch.append(config.LABELS_DATASET[index-1])
        
        # Return batch in form of dictionary 
        return {
            'index':np.array(indices),
            'lab_batch':np.array(lab_batch),
            'gray_batch':np.array(gray_batch),
            'labels':np.array(labels_batch)
        }

