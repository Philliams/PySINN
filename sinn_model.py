# SINN model with k-means initiation
from sklearn.cluster import KMeans
import numpy as np
from keras.layers import Input
from keras.models import Model
from sinn_layers import Metric, Shepard

class SINN_model:

	def __init__(self, input_matrix, output_matrix, num_of_clusters, nodes_per_cluster, batch_size, epochs, verbose=2, validation_split=0, shuffle_data=True, model_optimizer='RMSprop', loss_metric='mse'):

		# keep training parameters
		self.batch_size 		= batch_size
		self.epochs  			= epochs
		self.verbose 		  	= verbose
		self.validation_split 	= validation_split
		self.shuffle_data		= shuffle_data

		# build k-means clusters
		kmeans  = KMeans(n_clusters=num_of_clusters, random_state=0).fit(input_matrix)

		centers = kmeans.cluster_centers_
		labels  = kmeans.labels_

		# seperate labelled data by cluster, a maximum of nodes_per_cluster data points are kept for each cluster
		cluster_nodes = []
		cluster_topology = []

		for i in range(num_of_clusters):
			cluster_nodes.append([])
			cluster_topology.append([])

		for i in range(len(input_matrix)):

			index = labels[i]

			if(len(cluster_nodes[index]) < nodes_per_cluster):
				cluster_nodes[index].append(input_matrix[i])
				cluster_topology[index].append(output_matrix[i])

			if(self.is_matrix_valid(nodes_per_cluster,cluster_nodes)):
				break

		metric_initiation  = []
		shepard_initiation = []

		for i in range(len(cluster_nodes)):
			for j in range( min(len(cluster_nodes[i]),nodes_per_cluster)):
				metric_initiation.append(cluster_nodes[i][j])
				shepard_initiation.append(cluster_topology[i][j])

		print("----------------")
		print(metric_initiation)
		print("----------------")
		print(shepard_initiation)
		print("----------------")

		# build model from diverse dataset picked using k-means
		input_size = len(input_matrix[0])
		input_layer = Input(shape=(input_size,))

		# metric layer
		m = Metric(metric_initiation)(input_layer)

		# shepard layer
		s = Shepard(shepard_initiation)(m)

		# create model
		self.model = Model(input=input_layer, output=s)
		self.model.compile(optimizer=model_optimizer,loss=loss_metric)

	def predict(self, input_matrix):

		return self.model.predict_on_batch(input_matrix)

	def train_model(self, input_matrix, output_matrix):

		self.model.fit( input_matrix, output_matrix , batch_size=self.batch_size, nb_epoch=self.epochs, verbose=self.verbose, validation_split=self.validation_split, shuffle=self.shuffle_data)

	def is_matrix_valid(self,k,m):

		# check if matrix is not jagged (i.e. rectangular or square)
		for i in range(len(m)):
			if(len(m[i]) != k):
				return False

		return True



