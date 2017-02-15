# SINN_model_test.py

from sinn_model import SINN_model
from keras.datasets import mnist
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

data_training   = []
labels_training = []

data_testing   = []
labels_testing = []

print('starting...')

for i in range(len(X_train)):

	data_training.append( X_train[i].flatten('C'))

	classification = Y_train[i]

	label = np.zeros(10)
	label[classification] = 1

	labels_training.append(label)

print('done processing training data...')

for i in range(len(X_test)):

	data_testing.append( X_test[i].flatten())

	labels_testing.append(Y_test[i])

print('done processing testing data...')

data_training = np.array(data_training)
labels_training = np.array(labels_training)

data_testing = np.array(data_testing)
labels_testing = np.array(labels_testing)

s = SINN_model(data_training,labels_training,10,1,500,50,verbose=1,model_optimizer='adagrad',loss_metric='categorical_crossentropy')

s.train_model(data_training,labels_training)


