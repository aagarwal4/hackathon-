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

# load the user configs
with open('/conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]

# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
image_size = (299, 299)

print ("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
  cur_path = train_path + "/" + label
  count = 1
  for image_path in glob.glob(cur_path + "/*.jpg"):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    labels.append(label)
    #print ("[INFO] processed - " + str(count))
    count += 1
  print ("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
#print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
  json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")

print ("[STATUS] features and labels saved..")

# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

#Training Machine Learning algorithm
# organize imports
from __future__ import print_function

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

# load the user configs
with open('/conf/conf.json') as f:    
  config = json.load(f)

# config variables
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and validation data
(trainData, valData, trainLabels, valLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] val data   : {}".format(valData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] val labels : {}".format(valLabels.shape))

# use SVM as the model
print ("[INFO] creating model...")

classifier = SVC(probability=True)

classifier.fit(trainData, trainLabels)

# use rank-1 and rank-3 predictions
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_3 = 0

# loop over test data
for (label, features) in zip(valLabels, valData):
  # predict the probability of each class label and
  # take the top-5 class labels
  predictions = classifier.predict_proba(np.atleast_2d(features))[0]
  predictions = np.argsort(predictions)[::-1][:5]

  # rank-1 prediction increment
  if label == predictions[0]:
    rank_1 += 1

  # rank-3 prediction increment
  if label in predictions[:3]:
    rank_3 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(valLabels))) * 100
rank_3 = (rank_3 / float(len(valLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-3: {:.2f}%\n\n".format(rank_3))


# evaluate the model of train data
preds_train = classifier.predict(trainData)

# evaluate the model of test data
preds_val = classifier.predict(valData)

# write the classification report to file
f.write("{}\n".format(classification_report(valLabels, preds_val)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(classifier, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(valLabels, preds_val)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()

train_accuracy = accuracy_score(trainLabels, preds_train)
with open('train.txt', 'w') as f:
    f.write("The training accuracy is:" + str(train_accuracy) + '\n\n')
#     f.write("The actual labels are:" + str([train_labels[idx] for idx in trainLabels]) + '\n\n')
#     f.write("The predicted labels are:" + str([train_labels[idx] for idx in preds_train]) + '\n\n')

val_accuracy = accuracy_score(valLabels, preds_val)
with open('val.txt', 'w') as f:
    f.write("The validation accuracy is:" + str(val_accuracy) + '\n\n')
#     f.write("The actual labels are:" + str([train_labels[idx] for idx in valLabels]) + '\n\n')
#     f.write("The predicted labels are:" + str([train_labels[idx] for idx in preds_val]) + '\n\n')