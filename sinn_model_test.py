# SINN_model_test.py

import random
from sinn_model import SINN_model

x_possible = [-1,1]
y_possible = [-1,1]
# z_possible = [-2,-1,1,2]

test_data   = []
test_labels = []

for i in range(100):

	x = random.choice(x_possible)
	y = random.choice(y_possible)
	z = x * y

	x += random.uniform(-0.1,0.1)
	y += random.uniform(-0.1,0.1)
	z += random.uniform(-0.1,0.1)

	test_data.append([x,y])
	test_labels.append([z])

s = SINN_model(test_data,test_labels,4,4,100,10)
s.train_model(test_data,test_labels)

test_arr = [[1,1],[-1,-1]]

print(s.predict(test_arr))
