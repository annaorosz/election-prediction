
# coding: utf-8

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



# In[11]:




# In[25]:

testfile = "test_potus_by_county.csv"
reader = np.genfromtxt(testfile, delimiter=',')
x_predict = reader[1:,:]

m = load_model('my_model.h5')

y_predict = m.predict(x_predict, batch_size=100, verbose=1)


# In[29]:

#print out the number of Obamas
print(len([x for x in y_predict if x<=0.5]))


# In[30]:

file = open("predictions.csv", "w")
file.write("Winner\n")
for x in np.nditer(y_predict):
    if x > 0.5: 
        file.write("Mitt Romney\n") 
    else: 
        file.write("Barack Obama\n")


# In[ ]:



