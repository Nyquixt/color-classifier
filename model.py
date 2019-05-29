from __future__ import division
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import numpy as np 
import pickle

X = pickle.load(open('X.pickle', 'rb'))
Y = pickle.load(open('Y.pickle', 'rb'))

X = X / 255.0

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1:]))) # (3,)
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('softmax'))

tensorboard = TensorBoard(log_dir='logs/')

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.03),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X, Y, 
        validation_split=0.15, 
        epochs=100,
        verbose=1,
        callbacks=[tensorboard])

model.save('colorClassifier.model')
