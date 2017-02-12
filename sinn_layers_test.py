# SINN layers test file
from keras import backend as K
from sinn_layers import Metric, Shepard
from keras.layers import Input
from keras.models import Model
import numpy as np
from math import exp
import random

# ///////////// metric testing code ////////////////

test_arr = np.array([[1,1,1],[2,2,2],[3,3,3]],dtype='float32')

predict_arr = np.array([[1,1,1],[0,0,0],[1,1,1]],dtype='float32')

input_arr = np.array([[1,1,1],[2,2,2],[3,3,3]],dtype='float32')

inputs = Input(shape=(3,))
m = Metric(test_arr)(inputs)

model = Model(input=inputs, output=m)

model.compile(optimizer='adagrad',loss='mse')

# model.fit( input_arr, predict_arr , batch_size=3, nb_epoch=5000, verbose=2, validation_split=0, shuffle=True)

res = model.predict_on_batch(test_arr)

print(res)

# ///////////// shepard testing code ////////////////

input_arr = np.array([[2,1,0],[1,2,0],[1,1,0]])

output_arr = np.array([[1,-1],[2,-2],[-1,1]])

inputs = Input(shape=(3,))
s = Shepard(output_arr)(inputs)

model = Model(input=inputs, output=s)

model.compile(optimizer='adagrad',loss='mse')

res = model.predict_on_batch(input_arr)

print(res)

# ///////////// integration testing code ////////////////

K.set_epsilon(1e-3)

input_arr = []
output_arr = []

n = 15

for i in range(n):
    for j in range(n):
        x = i*4/n - 2
        y = j*4/n - 2
        z = exp(-y*y-x*x)

        input_arr.append([x,y])
        output_arr.append([z])

input_arr = np.array(input_arr)
output_arr = np.array(output_arr)

inputs = Input(shape=(2,))
m = Metric(input_arr)(inputs)
s = Shepard(output_arr)(m)

model = Model(input=inputs, output=s)

model.compile(optimizer='RMSprop',loss='mae')

input_arr = []
output_arr = []

for i in range(1000):

    x = random.uniform(-2, 2)
    y = random.uniform(-2, 2)
    z = exp(-y*y-x*x)
    input_arr.append([x,y])
    output_arr.append([z])

input_arr = np.array(input_arr)
output_arr = np.array(output_arr)

model.fit( input_arr, output_arr , batch_size=1000, nb_epoch=5, verbose=2, validation_split=0, shuffle=True)

input_arr = []
output_arr = []

for i in range(25):

    x = random.uniform(-2, 2)
    y = random.uniform(-2, 2)
    z = exp(-y*y-x*x)

    input_arr.append([x,y])
    output_arr.append([z])

input_arr = np.array(input_arr)
output_arr = np.array(output_arr)

print(model.predict_on_batch(input_arr))
print(output_arr)



