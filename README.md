***************************************
Dependencies & decisions about my model
***************************************

Software Dependencies:
Keras
Tensorflow
Python 3

------------------------------------------------------------------------------------------------------------------------

Why I chose to use these softwares and model:

In Keras, a neural network model can be implemented efficiently and can be used with multiple different backends:
Tensorflow, Theano, etc which all have neural network modules.

I chose to use a Sequential neural network model which has a stack of layers, because I believe that this gives a
sophisticated, yet not too complicated model to predict the presidential winners.

This model is proficient because it can give better classification by using non-linear boundaries and it is easy
to prevent overfitting (contrary to a decision tree, for example).
There were only 2 possible outputs, so a more complicated neural network was not necessary in this case.
By making adjustments to the model, such as BathNormalization, Dropouts, shuffling the data, etc.,
the performance was further improved.

With a neural network, it is much easier to adjust some settings (compared to other ML models),
most importantly, normalizing the data of features and accounting for skewed data.
(Further explained in performance.txt)

********************************************************************************
Data about the expected performance of my model & how I evaluated the best model
********************************************************************************

Statistics at Epoch 10000/10000:
loss: 0.1911, acc: 0.7397, val_loss: 0.1664, val_acc: 0.86

------------------------------------------------------------------------------------------------------------------------

Possibly the most important adjustment I made was to change the class-weight for each output. I noticed that in the
training data one output was too overrepresented. (Mitt Romney was the output for 78% of the data). After modifying the
model's class weights, the accuracy increased by 10%.

The dropouts also allowed significant improvement (by several %). This is because with using Dropouts, I was able to
avoid overfitting.

It was also important to incorporate BatchNormalization when I noticed that the data was very different among different
features. For example, average household size was never over 10, but the population was always in the thousands. Without
BatchNormalization, it would be difficult to compare these values and build an efficient model over it.

I also set differing number of units when calling Dense() on each layer.
This adjusted the dimension of the output. I noticed that when I steadily increase, then decrease the number of units
across my layers the accuracy improved.

Other adjustments that I made along the way were the learning speed, choosing relu for most layers and
vs sigmoid for my output layer and shuffling the rows of data.

Through these changes and adjustments, my validation accuracy went up to around 85% with 10000 epochs.
It is important to note that the training accuracy stayed at around 73%. This is due to the Droputs.
Therefore, it is a good sign that the validation accuracy is higher than the training accuracy:
this proves that Dropouts alone improve the accuracy by several %.

