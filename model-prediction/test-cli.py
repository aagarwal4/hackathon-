# organize imports
from __future__ import print_function

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#imports
import numpy as np
import pickle
import sys

# keras imports
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras import backend as K

# S3 imports
from io import BytesIO
import boto3, os

image_path = sys.argv[1]

# load the trained SVM classifier
print ("[INFO] loading the classifier...")
with open("../../rootkey.csv") as f:
    ACCESS_ID = f.readline().strip().split('=')[1]
    ACCESS_KEY = f.readline().strip().split('=')[1]
    
s3 = boto3.resource('s3', 
                    aws_access_key_id=ACCESS_ID,
                    aws_secret_access_key= ACCESS_KEY)

myBucket = s3.Bucket('hackathon-nissan')

with BytesIO() as data:
    myBucket.download_fileobj("classifier-models.pickle", data)
    data.seek(0)    # move back to the beginning after writing
    classifier = pickle.load(data)
    
# get all the train labels
train_labels = ["SUNNY-XL", "MICRA-PETROL", "TERRANO-P", "TERRANO-PRIME", "REDIGO"]    
base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
image_size = (299, 299)
img = image.load_img(image_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print('Preprocessing image.........')
x = preprocess_input(x)
print('Predicting.........')
feature = model.predict(x)
flat = feature.flatten()
flat = np.expand_dims(flat, axis=0)
print('Classifying.........')
preds = classifier.predict(flat)
prediction = train_labels[preds[0]]
print('Prediction done.........')
K.clear_session()
print(prediction)
