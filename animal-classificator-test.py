import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import warnings
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import (Activation, Dropout, Flatten, Dense, Input, Conv2D, 
                          MaxPooling2D, BatchNormalization, Concatenate,
                          Conv2DTranspose, concatenate)
from keras.models import Sequential, model_from_json
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import Sequence

from numpy import argmax
from PIL import Image
from tqdm import tqdm

from AnimalGenerator import AnimalGenerator

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print('Model uploaded!')

loaded_model.load_weights('model.h5')
print('Weights uploaded!')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy')
print('LabelEncoder uploaded!')

BATCH_SIZE = 32
SHAPE = (128,128,3)

def translator(x):
    return {
        'cane': 'Perro',
        'cavallo': 'Caballo',
        'elefante': 'Elefante',
        'farfalla': 'Mariposa',
        'gallina': 'Gallina',
        'gatto': 'Gato',
        'mucca': 'Vaca',
        'pecora': 'Oveja',
        'ragno': 'Araña',
        'scoiattolo': 'Ardilla',
    }[x]

paths = ['image.jpg']
labels = ['']

paths = np.array(paths)
labels = np.array(labels)

animal_gen = AnimalGenerator(paths, labels, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)

animal_pred = loaded_model.predict_generator(animal_gen)

# invert first example
inverted = label_encoder.inverse_transform([argmax(animal_pred[0])])

print('Perro:    {:6.2f} %'.format(animal_pred[0][0]*100))
print('Caballo:  {:6.2f} %'.format(animal_pred[0][1]*100))
print('Elefante: {:6.2f} %'.format(animal_pred[0][2]*100))
print('Mariposa: {:6.2f} %'.format(animal_pred[0][3]*100))
print('Gallina:  {:6.2f} %'.format(animal_pred[0][4]*100))
print('Gato:     {:6.2f} %'.format(animal_pred[0][5]*100))
print('Vaca:     {:6.2f} %'.format(animal_pred[0][6]*100))
print('Oveja:    {:6.2f} %'.format(animal_pred[0][7]*100))
print('Araña:    {:6.2f} %'.format(animal_pred[0][8]*100))
print('Ardilla:  {:6.2f} %'.format(animal_pred[0][9]*100))