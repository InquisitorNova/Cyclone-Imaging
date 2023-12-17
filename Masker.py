"""
Author: InquisitorNova(Kaylen Smith Darnbrook)
Date: 12/12/2023
File_Name: Tropical_Cyclone_Preprocessing
Description:
"""
import numpy as np 
import scipy as sp 
import sklearn as sk
import matplotlib.pyplot as plt 
import pandas as pd 
import pytorch_lightning as pyl
import torch as tor 
import seaborn as sns
from skimage.transform import resize
from PIL import Image 
from tensorflow import keras
from tensorflow.keras import layers
from skimage.transform import resize
import tensorflow as tf
import skorch
import tables
import h5py
import pickle

Cyclone_images = np.load(r"C:\Users\kdarn\OneDrive\Documents\Life's Portfolio\Projects\Machine Learning Personal Projects\Cyclone Imaging\FIltered_Images\Filtered_Cyclone_Images.npy", allow_pickle = True)
Labels = np.load(r"C:\Users\kdarn\OneDrive\Documents\Life's Portfolio\Projects\Machine Learning Personal Projects\Cyclone Imaging\FIltered_Images\Filtered_Labels.npy", allow_pickle = True)

Cyclone_images = np.array([resize(x, (200,200)) for x in Cyclone_images])

class SEblock(tf.keras.layers.Layer):
    def __init__(self,units, bottleneck_units):
        super().__init__()
        
        # Define the SE Block Layers:
        self.Dense = tf.keras.layers.Dense(units, kernel_initializer = "glorot_uniform", activation = "sigmoid")
        self.Global_Average_Pool = tf.keras.layers.GlobalAveragePooling2D()
        self.Bottleneck = tf.keras.layers.Dense(bottleneck_units, kernel_initializer = "he_normal", activation = "selu")
        self.Reshape = tf.keras.layers.Reshape((1,1,units))
        
    def call(self, x):
        x = self.Global_Average_Pool(x)
        x = self.Bottleneck(x)
        x = self.Dense(x)
        x = self.Reshape(x)
        return x
    
class Residual_Block(tf.keras.layers.Layer):
    def __init__(self,filters, units, units_bottleneck):
        super().__init__()
        
        # Define Residual Block Layers
        self.Conv_1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = "same", kernel_initializer= "he_normal")
        self.Conv_2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3), padding = "same", kernel_initializer = "he_normal")
        #self.ffn_1 = tf.keras.layers.Dense(filters, activation = "selu", kernel_initializer = "lecun_normal", kernel_regularizer = "l1_l2")
        #self.ffn_2 = tf.keras.layers.Dense(filters, activation = "selu", kernel_initializer = "lecun_normal", kernel_regularizer = "l1_l2")
        self.Conv_Bypass = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1), padding = "same", strides = 1, kernel_initializer = "he_normal")
        self.Leaky_Relu = tf.keras.layers.LeakyReLU()
        self.Max_Pooling = tf.keras.layers.MaxPool2D(2,2)
        self.Batch_Norm_1 = tf.keras.layers.BatchNormalization()
        self.Batch_Norm_2 = tf.keras.layers.BatchNormalization()
        self.Batch_Norm_3 = tf.keras.layers.BatchNormalization()
        self.Dropout_1 = tf.keras.layers.Dropout(0.3)
        self.Dropout_2 = tf.keras.layers.Dropout(0.3)
        self.Dropout_3 = tf.keras.layers.Dropout(0.3)
        self.Add_Layer = tf.keras.layers.Add()
        self.Multiply_Layer = tf.keras.layers.Multiply()
        self.SE_block = SEblock(units, units_bottleneck)
        
    def call(self, x):
        d = self.Conv_Bypass(x)
        d = self.Batch_Norm_1(d)
        
        x = self.Conv_1(x)
        x = self.Leaky_Relu(x)
        x = self.Batch_Norm_2(x)
        
        x = self.Conv_2(x)
        x = self.Batch_Norm_3(x)
        
        y = self.SE_block(x)
        y = self.Multiply_Layer([x,y])
        x = self.Add_Layer([y,d])
        
        x = self.Leaky_Relu(x)
        x = self.Max_Pooling(x)
        return x
    
def RE_Net():
    inputs = tf.keras.layers.Input(shape = (200,200,1))
    x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", kernel_initializer= "he_normal")(inputs)
    x = Residual_Block(filters = 32, units = 32, units_bottleneck = 4)(x)
    x = Residual_Block(filters = 64, units = 64, units_bottleneck = 16)(x)
    x = Residual_Block(filters = 128, units = 128, units_bottleneck = 32)(x)
    x = Residual_Block(filters = 128, units = 128, units_bottleneck = 32)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation = "gelu", kernel_initializer = "he_normal", kernel_regularizer = "l1_l2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(100, activation = "gelu", kernel_initializer = "he_normal", kernel_regularizer = "l1_l2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(100, activation = "gelu", kernel_initializer = "he_normal", kernel_regularizer = "l1_l2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    out = tf.keras.layers.Dense(1, activation = "sigmoid", kernel_initializer = "glorot_uniform")(x)
    Model = tf.keras.Model(inputs = inputs, outputs = out)
    
    return Model
SE_Net = RE_Net()
print(SE_Net.summary())

import os
lists = os.listdir(r"C:\Users\kdarn\OneDrive\Documents\Life's Portfolio\Projects\Machine Learning Personal Projects\Cyclone Imaging\Checkpoints")
path = r"C:\Users\kdarn\OneDrive\Documents\Life's Portfolio\Projects\Machine Learning Personal Projects\Cyclone Imaging\Checkpoints"

SE_Net.load_weights(path)

Predictions = SE_Net.predict(Cyclone_images[:,:,:,2].reshape(-1,200,200,1))
print(Predictions.shape)

mask = [bool(x) for x in np.round(Predictions).reshape(-1,)]
print(Cyclone_images[mask].shape)

np.save("FiltsCleaned_Images.npy", Cyclone_images[mask])
np.save("FiltsCleaned_Labels.npy", Labels[mask])

