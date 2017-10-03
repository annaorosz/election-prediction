from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import csv
import numpy as np
import random as rn
import tensorflow as tf

testfile = "test_potus_by_county.csv"
reader = np.genfromtxt(testfile, delimiter=',')
x_predict = reader[1:,:]

#load model binary that was built in build_model
m = load_model('my_model.h5')

#predict winners based on the model
y_predict = m.predict(x_predict, batch_size=100, verbose=1)

#write the winners to precictions.csv
file = open("predictions.csv", "w")
file.write("Winner\n")
for x in np.nditer(y_predict):
    if x > 0.5: 
        file.write("Mitt Romney\n") 
    else: 
        file.write("Barack Obama\n")
