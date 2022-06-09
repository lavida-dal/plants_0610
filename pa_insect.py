import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import json
from PIL import Image
from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers import Dropout, Activation, Dense
# from keras.layers import Flatten, MaxPooling2D
# from keras.layers import Conv2D
# from keras.models import
# load_model
# from keras.optimizers import Adam
# # from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import OneHotEncoder

from keras_applications.resnet50 import ResNet50

import tensorflow as tf
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from random import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


#import cv2

from sklearn.model_selection import train_test_split
import plotly.express as px
# from plotly import offline
# offline.init_notebook_mode(connected = True)

file_path = "./노지 작물 해충 진단 이미지"

image_path1 = "./노지 작물 해충 진단 이미지/Training/[T원천]12.파_0.정상"
image_path2 = "./노지 작물 해충 진단 이미지/Training/[T원천]12.파_2.해충"

label_path1 = "./노지 작물 해충 진단 이미지/Training/[T라벨링]12.파_0.정상"
label_path2 = "./노지 작물 해충 진단 이미지/Training/[T라벨링]12.파_2.해충"

X = []
# for file in os

num = 0
for file in os.listdir(image_path1):
    path = image_path1 + "/" + file
    img = Image.open(path)
    img = img.resize((256,256))
    img_array = np.array(img)
    X.append(img_array)
    if num == 1000:
        break
    num+=1
    # print(i, " 추가 완료 - 구조:", img_array.shape) # 불러온 이미지의 차원 확인 (세로X가로X색)
    #print(img_array.T.shape) #축변경 (색X가로X세로)

num = 0
for file in os.listdir(image_path2):
    path = image_path2 + "/" + file
    img = Image.open(path)
    img = img.resize((256,256))
    img_array = np.array(img)
    X.append(img_array)
    if num == 1000:
        break
    num+=1
    # print(i, " 추가 완료 - 구조:", img_array.shape) # 불러온 이미지의 차원 확인 (세로X가로X색)
    #print(img_array.T.shape) #축변경 (색X가로X세로)


Y = []
num = 0
for file in os.listdir(label_path1):
    dir_path = label_path1 + "/" + file
    with open(dir_path, "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    if json_data["annotations"]["disease"] == 16:
        Y.append(0)
    elif json_data["annotations"]["disease"] == 17:
        Y.append(1)
    else:
        Y.append(2)
    if num == 1000:
        break
    num+=1


num = 0
for file in os.listdir(label_path2):
    dir_path = label_path2 + "/" + file
    with open(dir_path, "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    if json_data["annotations"]["disease"] == 16:
        Y.append(0)
    elif json_data["annotations"]["disease"] == 17:
        Y.append(1)
    else:
        Y.append(2)
    if num == 1000:
        break
    num+=1

print(Y.count(0))
print(Y.count(1))
print(Y.count(2))
X = np.array(X)
print(X.shape)

# X = np.array(X)
Y = np.array(Y)
print(Y.shape)


Y = to_categorical(Y)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,shuffle=42)

# xy = (X_train, X_test, Y_train, Y_test)
#
# np.save("./img_data.npy", xy)

# if not os.path.exists("./model/"):
#     os.mkdir('./model/')
# X_train, X_test, Y_train, Y_test = np.load('./data/img_data.npy', allow_pickle=True)
#
# categories = list(str(i) for i in range(20))
# EPOCHS = 30
# BS = 32
# INIT_LR = 1e-3
# n_classes = len(categories)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding = "same", input_shape = X_train.shape[1:], activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# def show_images(count, X, y):   #이미지 개수, X, y 인자로 받기
#   fig, axes = plt.subplots(count//4,4,figsize=(16, count))
#   for i, ind in enumerate(np.random.randint(0, X.shape[0], count)):
#     ax = axes[i//4][i%4]
#     ax.imshow(np.squeeze(X[ind]),cmap ='gray')  #중복 차원제거 squeeze?
#     ax.title.set_text(shapes[np.argmax(y[ind])])
#     ax.set_xticks([])
#     ax.set_yticks([])
# show_images(8, X_train, y_train)

# model = Sequential([
#     Conv2D(32,(3,3),strides = 2,input_shape = (256,256,3), activation = 'relu'),
#     MaxPooling2D(),
#     Conv2D(32, (3,3), activation = 'relu'),
#     MaxPooling2D(),
#     Conv2D(32, (3,3), activation = 'relu'),
#     MaxPooling2D(),
#     Conv2D(32, (3,3), activation = 'relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dropout(0.5),
#     Dense(128, activation = 'relu'),
#     Dropout(0.5),
#     Dense(64, activation = 'relu'),
#     Dropout(0.5),
#     Dense(4, activation = 'softmax')
# ])
# model.summary()
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X_train, Y_train, epochs = 100, validation_data=(X_test, Y_test))

model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=(256, 256, 3), pooling=max, classes=3,
                 backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_data=(X_test, Y_test))
history = model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_data=(X_test, Y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc=0)

plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc=0)

plt.show()

loss, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy is {}%'.format(accuracy*100))

preds = model.predict(X_test)
# show_images(16,X_test, preds)

