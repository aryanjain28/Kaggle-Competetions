import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pygame

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BATCH_SIZE = 100
EPOCHS = 10
WIDTH = 256
HEIGHT = 256

trainDir = os.path.join(os.path.dirname(__file__), 'images/train')
testDir = os.path.join(os.path.dirname(__file__), 'images/test')

cats = os.path.join(os.path.dirname(__file__), 'images/train/cats')
dogs = os.path.join(os.path.dirname(__file__), 'images/train/dogs')

imgGen = ImageDataGenerator(rescale=1./255)

trainData = imgGen.flow_from_directory(
    directory=trainDir,
    target_size=(WIDTH, HEIGHT),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# testData = imgGen.flow_from_directory(
#     directory=testDir,
#     target_size=(WIDTH, HEIGHT),
#     color_mode='rgb',
#     class_mode='binary',
#     batch_size=BATCH_SIZE
# )




