
# coding: utf-8

# <b>Deep neural network for excersize</b>

# In[1]:

from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import csv
import numpy as np
import random as rn
import tensorflow as tf


# In[2]:

# make Keras take only the required amount of memory from the GPU and not allow
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
set_session(tf.Session(config=config))


# In[3]:

trainfile = "train_potus_by_county.csv"
reader = np.genfromtxt(trainfile, delimiter=',')
x_train = reader[1:,:-1]

# create shuffled ids
ids = list(range(x_train.shape[0]))
rn.shuffle(ids)

x_train = x_train[ids]


# In[4]:

with open(trainfile) as f:
    reader = csv.reader(f)
    y_train = [1. if x[-1] == 'Mitt Romney' else 0. for x in list(reader)[1:]]
    y_train = np.array(y_train)
    y_train = y_train[ids]  # randomize


# In[ ]:

# normalize MO and BO training examples (removing MO training samples)
# DO NOT EXECUTE!!!
bo = len([x for x in y_train if x==0])
mr = len([x for x in y_train if x==1])

mr_only = [k for k,x in enumerate(y_train) if x == 1]
ids_to_keep = mr_only[(mr-bo):]
ob_only = [k for k,x in enumerate(y_train) if x == 0]
ids_to_keep.extend(ob_only)

rn.shuffle(ids_to_keep)

# these ids will stay
print(len(ids_to_keep))

x_train = x_train[ids_to_keep]
y_train = y_train[ids_to_keep]
print(len(x_train))
print(len(y_train))

print(y_train)


# In[5]:

# Single layer neural network model
model = Sequential()
model.add(Dense(32, input_dim=14))  # input_dim: num of features
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))


# In[6]:

# configure the learning process
model.compile(loss='binary_crossentropy',
              #loss='mse',
              optimizer=SGD(lr=0.1, decay=1e-6),
              #optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
              #optimizer='SGD',
              #optimizer='rmsprop',
              metrics=['accuracy'])


# In[7]:

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=2000, batch_size=1000, validation_split=0.1,
          class_weight={0:0.78, 1:0.22},
          #shuffle=True
         )


# In[ ]:

# evalutate the model on the test data
#score = model.evaluate(x_test, y_test, batch_size=128)


# In[8]:

testfile = "test_potus_by_county.csv"
reader = np.genfromtxt(testfile, delimiter=',')
x_predict = reader[1:,:]
# evalutate the model on the test data
y_predict = model.predict(x_predict, batch_size=100, verbose=1)
y_predict


# In[9]:

print(len([x for x in y_predict if x<0.5]))


# In[10]:

model.save('my_model.h5')


# In[ ]:

m2 = load_model('my_model.h5')


# In[ ]:

y_predict = m2.predict(x_predict, batch_size=100, verbose=1)
y_predict


# In[1]:

print(len([x for x in y_predict if x<0.5]))


# In[ ]:



