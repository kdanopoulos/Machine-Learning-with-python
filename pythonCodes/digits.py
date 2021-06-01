
import tensorflow as tf
import numpy as np
import keras as kr


(x_train,y_train),(x_test,y_test) = kr.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

def preprocess_images(imgs): # should work for both a single image and multiple images
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape # make sure images are 28x28 and single-channel (grayscale)
    return imgs / 255.0

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)


model = kr.Sequential()
model.add(kr.layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(kr.layers.Conv2D(64,(3,3),activation='relu'))
model.add(kr.layers.MaxPooling2D(pool_size=(2,2)))
model.add(kr.layers.Dropout(0.25))
model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(128,activation='relu'))
model.add(kr.layers.Dropout(0.5))
model.add(kr.layers.Dense(10,activation='softmax'))

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)

history = model.fit(x_train,y_train,epochs=5)

(loss,accu) = model.evaluate(x_test,y_test)

print('The accurancy is : ',accu)
print('The loss is : ',loss)