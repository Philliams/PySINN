# PySINN

PySINN is a python implementation of the [Shepard Interpolation Neural Network](http://link.springer.com/chapter/10.1007/978-3-319-50832-0_34) using Keras and Theano.

###### Dependencies:

PySINN uses [Keras](https://keras.io) and [Theano](http://deeplearning.net/software/theano/) for all of the Machine Learning computations and [Scikit Learn](http://scikit-learn.org/stable/) for the K-means clustering.

###### Usage:
```python
# create Shepard Interpolation Neural Network object
s = SINN_model(data_training,labels_training,number_of_clusters,nodes_per_cluster,batch_size,epochs,verbose=1,model_optimizer='RMSprop',loss_metric='categorical_crossentropy')

# Train the model
s.train_model(data_training,labels_training,data_testing,labels_testing)

# Use model
predictions = s.predict(data_testing)
```

