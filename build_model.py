from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import csv
import numpy as np
import random as rn
import tensorflow as tf

trainfile = "train_potus_by_county.csv"
reader = np.genfromtxt(trainfile, delimiter=',')
x_train = reader[1:,:-1]

#shuffle ids
ids = list(range(x_train.shape[0]))
rn.shuffle(ids)

x_train = x_train[ids]

with open(trainfile) as f:
    reader = csv.reader(f)
    y_train = [1. if x[-1] == 'Mitt Romney' else 0. for x in list(reader)[1:]]
    y_train = np.array(y_train)
    y_train = y_train[ids]  # randomize

#layers of the model
model = Sequential()
#first layer
model.add(Dense(32, input_dim=14))  # input_dim: num of features
model.add(BatchNormalization())
model.add(Activation('relu'))
#hidden layers
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
#output layer
model.add(Dense(1, activation='sigmoid'))

# configure the learning process
model.compile(loss='binary_crossentropy',
          optimizer=SGD(lr=0.1, decay=1e-6),
          metrics=['accuracy'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=10000, batch_size=1000, validation_split=0.1,
      class_weight={0:0.78, 1:0.22},
     )

model.save('my_model.h5')
