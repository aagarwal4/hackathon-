# organize imports
from __future__ import print_function

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# S3 imports
from io import BytesIO
import boto3, os

# load the user configs
with open('conf.json') as f:
    config = json.load(f)
    
# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
test_path = config["test_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
seed = config["seed"]
classifier_path = config["classifier_path"]

# load the trained SVM classifier
print ("[INFO] loading the classifier...")
#with open("../../rootkey.csv") as f:
#    ACCESS_ID = f.readline().strip().split('=')[1]
#    ACCESS_KEY = f.readline().strip().split('=')[1]
    
#s3 = boto3.resource('s3', 
#                    aws_access_key_id=ACCESS_ID,
#                    aws_secret_access_key= ACCESS_KEY)

#myBucket = s3.Bucket('hackathon-nissan')

#with BytesIO() as data:
#    myBucket.download_fileobj("classifier-models.pickle", data)
#    data.seek(0)    # move back to the beginning after writing
#    classifier = pickle.load(data)

classifier = pickle.load(open("classifier.pickle", "rb"))   
base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
image_size = (299, 299)

# get all the test labels
train_labels = ['MICRA-PETROL', 'REDIGO', 'SUNNY-XL', 'TERRANO-P', 'TERRANO-PRIME']

test_preds = []
testLabels = []
# loop over all the labels in the folder
for i, label in enumerate(train_labels):
  cur_path = test_path + "/" + label
  for image_path in glob.glob(cur_path + "/*.jpg"):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    prediction = train_labels[preds[0]]
    test_preds.append(image_path.split("/")[-1] + ' ' + prediction)
    testLabels.append(image_path.split("/")[-1] + ' ' + label)
  for image_path in glob.glob(cur_path + "/*.png"):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    test_preds.append(image_path.split("/")[-1] + ' ' + train_labels[preds[0]])
    testLabels.append(image_path.split("/")[-1] + ' ' + label)
    
test_accuracy = accuracy_score(testLabels, test_preds)
with open('test.txt', 'w') as f:
    f.write("The test accuracy is:" + str(test_accuracy) + '\n\n')
#     f.write("The actual labels are:" + str(testLabels) + '\n\n')
#     f.write("The predicted labels are:" + str(test_preds) + '\n')
