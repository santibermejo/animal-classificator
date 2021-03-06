import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import warnings

from AnimalGenerator import AnimalGenerator

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import (Activation, Dropout, Flatten, Dense, Input, Conv2D, 
                          MaxPooling2D, BatchNormalization, Concatenate,
                          Conv2DTranspose, concatenate)
from keras.models import Sequential
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from numpy import argmax
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 64
SHAPE = (128,128,3)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def getTrainDataset():
  paths = []
  labels = []
  
  path_to_train = os.path.join(os.getcwd(), 'raw-img')
  
  animals_list = os.listdir(path_to_train) 
  animals_qty = len(animals_list)

  for animal_index in range(animals_qty):
    animal = animals_list[animal_index]
    animal_path = os.path.join(path_to_train, animal)
    images_list = os.listdir(animal_path)
    images_qty = len(images_list)

    for image_index in range(images_qty):
      image = images_list[image_index]
      path = os.path.join(animal_path, image)
      paths.append(path)
      labels.append(animal)

  return np.array(paths), np.array(labels)

paths, labels = getTrainDataset()

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
animal_labels = onehot_encoder.fit_transform(integer_encoded)

def create_model(input_shape):
    
    dropRate = 0.25
    
    init = Input(input_shape)
    x = Conv2D(8, (3, 3), activation='relu')(init)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    c3 = Conv2D(16, (7, 7), activation='relu', padding='same')(x)
    c4 = Conv2D(16, (1, 1), activation='relu', padding='same')(x)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)
    
    model = Model(init, x)
    
    return model

model = create_model(SHAPE)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

X_train, X_test, y_train, y_test = train_test_split(paths, animal_labels, test_size = 0.2, random_state=0)

tg = AnimalGenerator(X_train, y_train, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)
vg = AnimalGenerator(X_test, y_test, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)

epochs = 50

checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
early_stopping = EarlyStopping(patience=9, verbose=1)
use_multiprocessing = False
workers = 1

history = model.fit_generator(
    generator=tg,
    validation_data=vg,
    epochs=epochs,
    use_multiprocessing=use_multiprocessing,
    workers=workers,
    verbose=1,
    callbacks=[early_stopping, checkpoint, reduceLROnPlato])

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")

model_json = model.to_json()
with open('model.json', "w") as json_file:
  json_file.write(model_json)
print('Model saved!')
model.save_weights('model.h5')
print('Weights saved!')
np.save('label_encoder.npy', label_encoder.classes_)
print('LabelEncoder saved!')