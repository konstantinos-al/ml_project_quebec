from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
import xmltodict
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

xmls_dir = os.path.join(os.getcwd(), 'annotations')
imgs_labels = []

for dir_path in glob.glob(xmls_dir):
    for xml_path in glob.glob(os.path.join(dir_path, '*.xml')):
        with open (xml_path, 'rb') as my_xml:
            xmld = xmltodict.parse(my_xml)
            tags = xmld['annotation']

            if 'object' in tags:
                _object_ = tags['object']

                if type(_object_) == list:
                    od = _object_[0]
                    if od['name'] == 'With Helmet':
                        imgs_labels.append([0, os.path.join(os.getcwd(), str('images')+'\\'+tags['filename'])])
                    else:
                        imgs_labels.append([1, os.path.join(os.getcwd(), str('images')+'\\'+tags['filename'])])
                else:
                    if _object_['name'] == 'With Helmet':
                        imgs_labels.append([0, os.path.join(os.getcwd(), str('images')+'\\'+tags['filename'])])
                    else:
                        imgs_labels.append([1, os.path.join(os.getcwd(), str('images')+'\\'+tags['filename'])])



# 3 arxeia sta xmls den exoun label -> BRES POIES EIKONES EINAI
imgs_labels = np.array(imgs_labels).reshape(-1,2,1)
labels = imgs_labels[:,0].reshape(-1,1).astype(np.int32)
data = imgs_labels[:,1].flatten()
# print(data.shape)
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)
# print(labels)



imgs_dir = os.path.join(os.getcwd(), 'images')
imgs = []

for img_path in data:
    img = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
    imgs.append(img/255)

imgs = np.array(imgs)

max_height = max(map(len, imgs[0]))
max_width = max(map(len, imgs))
print(max_height, max_width)


images = []
for img in imgs:
    images.append(np.full((max_height, max_width, 3), (0,0,0), dtype='int32'))

images = np.array(images, dtype='int32')

trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)
# print(trainX.shape, testX.shape, trainY.shape, testY.shape)

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


num_classes = 2

trainY = to_categorical(trainY, num_classes)
trainY = np.array(trainY, dtype='int32')

testY = to_categorical(testY, num_classes)
testY = np.array(testY, dtype='int32')

print(trainX.shape, testX.shape, trainY.shape, testY.shape)

print(trainY)
INIT_LR = 1e-4
EPOCHS = 20
BS = 8


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(max_height, max_width, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
