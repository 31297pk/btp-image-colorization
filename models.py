
# coding: utf-8

# In[1]:


import tensorflow as tf
# import tensorboard as board
import tensorflow.layers as layers
import keras
import tensorflow.nn as nn
import numpy as np
import keras.models as models


# In[2]:


import time as tm
import config_file as config


# In[3]:


# Low level feature extraction layer
########################
"""Classifier network"""
########################
class Classifier():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.denseNet_1 = layers.Dense(256, activation=nn.sigmoid)
        self.denseNet_2 = layers.Dense(self.num_classes, activation=nn.sigmoid)
        self.softmax_layer = nn.softmax
        return
    
    def forward(self, inputs):
        features = self.denseNet_1(inputs)
#         print(features)
        features = self.denseNet_2(features)
#         print(features)
        features = self.softmax_layer(features)
#         print(features)
        return features
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[4]:


#############################
"""Low-level network"""
#############################
class LowLevelFeatureExtractor():
    def __init__(self):
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=[3,3], strides=[2,2], padding='same', activation=nn.sigmoid)
        self.conv_2 = layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_3 = layers.Conv2D(filters=128, kernel_size=[3,3], strides=[2,2], padding='same', activation=nn.sigmoid)
        self.conv_4 = layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_5 = layers.Conv2D(filters=256, kernel_size=[3,3], strides=[2,2], padding='same', activation=nn.sigmoid)
        self.conv_6 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        
        return
    def forward(self, inputs):
        features = self.conv_1(inputs)
#         print(features)
        features = self.conv_2(features)
#         print(features)
        features = self.conv_3(features)
#         print(features)
        features = self.conv_4(features)
#         print(features)
        features = self.conv_5(features)
#         print(features)
        features = self.conv_6(features)
#         print(features)
        return features
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[5]:


##############################
"""Mid level network"""
##############################
class MidLevelFeaturesExtractor():
    def __init__(self):
        self.conv_1 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_2 = layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        
        return
    
    def forward(self, inputs):
        features = self.conv_1(inputs)
#         print(features)
        features = self.conv_2(features)
#         print(features)
        
        return features
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[6]:


###########################
"""Global features network"""
##########################
class GlobalFeaturesExtractor():
    def __init__(self):
        self.conv_1 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[2,2], padding='same', activation=nn.sigmoid)
        self.conv_2 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_3 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[2,2], padding='same', activation=nn.sigmoid)
        self.conv_4 = layers.Conv2D(filters=512, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        
        self.flatten = layers.Flatten()
        
        self.dense_5 = layers.Dense(1024, activation=nn.sigmoid)
        self.dense_6 = layers.Dense(512, activation=nn.sigmoid)
        self.dense_7 = layers.Dense(256, activation=nn.sigmoid)
        
        return        
        
    def forward(self, inputs):
        features = self.conv_1(inputs)
#         print(features)
        features = self.conv_2(features)
#         print(features)
        features = self.conv_3(features)
#         print(features)
        features = self.conv_4(features)
        
#         print(features)
        features = self.flatten(features)
        
#         print(features)
        features = self.dense_5(features)
#         print(features)
        features = self.dense_6(features)
#         print(features)
        features = self.dense_7(features)
#         print(features)
        
        return features
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[7]:


##############################
"""Merging module"""
##############################

def fuseLayers(ll_features, gl_features):
    shape_list = ll_features.shape.as_list()
    y = gl_features
    y = tf.reshape(y, [-1,1,1,256])
    y = tf.concat([y]*shape_list[1], 1)
    y = tf.concat([y]*shape_list[2], 2)
    y = tf.concat([ll_features, y], 3)
    
    return y


# In[8]:


##########################
"""Colorization network"""
##########################
class ColorizedImageExtractor():
    def __init__(self):
        self.conv_1 = layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.upsample_1 = tf.image.resize_images
        
        self.conv_2 = layers.Conv2D(filters=64, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_3 = layers.Conv2D(filters=64, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.upsample_2 = tf.image.resize_images
        
        self.conv_4 = layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.conv_5 = layers.Conv2D(filters=2, kernel_size=[3,3], strides=[1,1], padding='same', activation=nn.sigmoid)
        self.upsample_3 = tf.image.resize_images
        
        return        
        
    def forward(self, inputs):
        features = self.conv_1(inputs)
#         print(features)
        features = self.upsample_1(features, size=(64, 64))
#         print(features)

        features = self.conv_2(features)
#         print(features)
        features = self.conv_3(features)
#         print(features)
        features = self.upsample_1(features, size=(128, 128))
#         print(features)
        
        features = self.conv_4(features)
#         print(features)
        features = self.conv_5(features)
#         print(features)
        features = self.upsample_1(features, size=(256, 256))
#         print(features)
        
        return features
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[9]:


###################################################
#########  Colorizaion class  ###################
###################################################
class ColorizationNetwork():
    def __init__(self, num_labels):
        
        self.num_labels = num_labels
        
        self.local_1 = LowLevelFeatureExtractor()
        self.local_2 = LowLevelFeatureExtractor()
        
        self.global_3 = GlobalFeaturesExtractor()
        self.mid_level_4 = MidLevelFeaturesExtractor()
        
        self.classifier_5 = Classifier(self.num_labels)
        self.colorization_6 = ColorizedImageExtractor()
        
        return
    
    def forward(self, inputs):
#         print('Extracting local features: ')
        features_l = self.local_1(inputs)
        features_g = self.local_2(inputs)
        
#         print("Extracting mid-level features: ")
        features_l = self.mid_level_4(features_l)
#         print("Extracting global features: ")
        features_g = self.global_3(features_g)
#         print("Extracting labels: ")
        labels = self.classifier_5(features_g)
        
#         print("Fusing layers: ")
        features = fuseLayers(features_l, features_g)
#         print(features)
#         print("ColorizationNetwork running: ")
        features = self.colorization_6(features)
        
        return features, labels
    
    def __call__(self, inputs):
        return self.forward(inputs)
        
        

