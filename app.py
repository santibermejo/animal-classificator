from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
from AnimalGenerator import AnimalGenerator
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from numpy import argmax

import numpy as np
import keras.models
import sys 
import os
import cv2

#initalize our flask app
app = Flask(__name__)

json_file = open('load/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print('Model uploaded!')

loaded_model.load_weights('load/model.h5')
print('Weights uploaded!')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('load/label_encoder.npy')
print('LabelEncoder uploaded!')

BATCH_SIZE = 32
SHAPE = (128,128,3)
UPLOAD_FOLDER = os.getcwd()

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
	file = request.files['image_file']
	filename = 'images/'+file.filename.split('.')[0]+'.'+file.filename.split('.')[-1]
	file.save(os.path.join(UPLOAD_FOLDER,filename))
	
	image_path = os.path.join(UPLOAD_FOLDER,filename)
	paths 	= [image_path]
	labels 	= ['']

	paths = np.array(paths)
	labels = np.array(labels)

	animal_gen = AnimalGenerator(paths, labels, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)

	animal_pred = loaded_model.predict_generator(animal_gen)

	inverted = label_encoder.inverse_transform([argmax(animal_pred[0])])

	perro 		= 'Perro:    {:6.2f} %'.format(animal_pred[0][0]*100)
	caballo 	= 'Caballo:  {:6.2f} %'.format(animal_pred[0][1]*100)
	elefante 	= 'Elefante: {:6.2f} %'.format(animal_pred[0][2]*100)
	mariposa 	= 'Mariposa: {:6.2f} %'.format(animal_pred[0][3]*100)
	gallina 	= 'Gallina:  {:6.2f} %'.format(animal_pred[0][4]*100)
	gato 		= 'Gato:     {:6.2f} %'.format(animal_pred[0][5]*100)
	vaca 		= 'Vaca:     {:6.2f} %'.format(animal_pred[0][6]*100)
	oveja 		= 'Oveja:    {:6.2f} %'.format(animal_pred[0][7]*100)
	arana 		= 'Araña:    {:6.2f} %'.format(animal_pred[0][8]*100)
	ardilla 	= 'Ardilla:  {:6.2f} %'.format(animal_pred[0][9]*100)

	results = ( perro + ' <br /> ' + 
				caballo + ' <br /> ' +
				elefante + ' <br /> ' +
				mariposa + ' <br /> ' +
				gallina + ' <br /> ' +
				gato + ' <br /> ' +
				vaca + ' <br /> ' +
				oveja + ' <br /> ' +
				arana + ' <br /> ' +
				ardilla )

	return results


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5010))
	app.run(host='0.0.0.0', port=port)

