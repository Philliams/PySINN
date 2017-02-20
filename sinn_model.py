# SINN model with k-means initiation
from sklearn.cluster import KMeans
import numpy as np
from keras.layers import Input, Activation, Dropout
from keras.models import Model
from sinn_layers import Metric, Shepard

class SINN_model:

	def __init__(self, input_matrix, output_matrix, num_of_clusters, nodes_per_cluster, batch_size, epochs, mode='classification', verbose=2, validation_split=0.05, shuffle_data=True, model_optimizer='RMSprop', loss_metric='mse', initiation_mode='k_means'):

		# keep training parameters
		self.batch_size 		= batch_size
		self.epochs  			= epochs
		self.verbose 		  	= verbose
		self.validation_split 	= validation_split
		self.shuffle_data		= shuffle_data


		(separated_data,separated_labels) = self.separate_data_by_class(input_matrix,output_matrix)

		( initiation_data , initiation_labels ) = ([],[])

		if( initiation_mode == 'k_means'):
			(initiation_data , initiation_labels) = self.k_means_initiation(separated_data,separated_labels,num_of_clusters,nodes_per_cluster)
		elif (initiation_mode == 'distributed'):
			(initiation_data,initiation_labels) = self.distributed_initiation(separated_data,separated_labels,num_of_clusters*nodes_per_cluster)

		print('number of encoded nodes : ' + str(len(initiation_data)))

		# build model from diverse dataset picked using k-means
		input_size = len(input_matrix[0])
		input_layer = Input(shape=(input_size,))

		# metric layer
		m = Metric(initiation_data)(input_layer)
		# d = Dropout(0.2)(m)
		# shepard layer
		s = Shepard(initiation_labels)(m)

		if(mode == 'classification'):

			# softmax = Dense(len(output_matrix[0]), activation='softmax')(s)
			softmax = Activation('softmax')(s)
			self.model = Model(input=input_layer, output=softmax)
			self.model.compile(optimizer=model_optimizer,loss=loss_metric,metrics=['categorical_accuracy'])

		else:

			self.model = Model(input=input_layer, output=s)
			self.model.compile(optimizer=model_optimizer,loss=loss_metric)

		print('model complete...')

	def k_means_initiation(self,separated_data,separated_labels,num_of_clusters,nodes_per_cluster):

		initiation_data   = []
		initiation_labels = []
		
		for i in range(len(separated_data)):

			(metric_initiation,shepard_initiation) = self.get_clusters(num_of_clusters,nodes_per_cluster,separated_data[i],separated_labels[i])

			initiation_data   += metric_initiation
			initiation_labels += shepard_initiation

		return (initiation_data,initiation_labels)


	def distributed_initiation(self,separated_data,separated_labels,nodes_per_class):

		initiation_data   = []
		initiation_labels = []

		for i in range(len(separated_data)):

			for j in range( min( nodes_per_class , len(separated_data[i]) )):

				initiation_data.append(separated_data[i][j])
				initiation_labels.append(separated_labels[i][j])

		return (initiation_data,initiation_labels)


	def get_clusters(self,num_of_clusters,nodes_per_cluster,input_matrix,output_matrix):
		# build k-means clusters
		print('building k-means clusters...')
		kmeans  = KMeans(n_clusters=num_of_clusters, random_state=0, max_iter=1000,n_init=20).fit(input_matrix)
		print('done...')
		centers = kmeans.cluster_centers_
		labels  = kmeans.labels_

		# seperate labelled data by cluster, a maximum of nodes_per_cluster data points are kept for each cluster
		cluster_nodes = []
		cluster_topology = []

		print('separating clusters...')

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

		print('done...')
		print('building initiation matrices...')
		metric_initiation  = []
		shepard_initiation = []

		for i in range(len(cluster_nodes)):
			for j in range( min(len(cluster_nodes[i]),nodes_per_cluster)):
				metric_initiation.append(cluster_nodes[i][j])
				shepard_initiation.append(cluster_topology[i][j])

		return (metric_initiation,shepard_initiation)

	def separate_data_by_class(self, input_matrix, output_matrix):

		separated_data   = []
		separated_labels = []

		for i in range(len(output_matrix[0])):
			separated_data.append([])
			separated_labels.append([])

		for i in range(len(input_matrix)):

			index = np.argmax(output_matrix[i])

			separated_data[index].append(input_matrix[i])
			separated_labels[index].append(output_matrix[i])

		return (separated_data,separated_labels)

	def predict(self, input_matrix):

		return self.model.predict_on_batch(input_matrix)

	def train_model(self, input_matrix, output_matrix, input_test=None, output_test=None):

		print('training starting...')

		if( input_test != None):
			self.model.fit( input_matrix, output_matrix , batch_size=self.batch_size, nb_epoch=self.epochs, verbose=self.verbose, validation_data=(input_test,output_test), shuffle=self.shuffle_data)
		else:
			self.model.fit( input_matrix, output_matrix , batch_size=self.batch_size, nb_epoch=self.epochs, verbose=self.verbose, validation_split=self.validation_split, shuffle=self.shuffle_data)

	def evaluate_model(self,input_matrix,output_matrix):

		return self.model.evaluate(input_matrix, output_matrix, batch_size=self.batch_size, verbose=self.verbose)

	def is_matrix_valid(self,k,m):

		# check if matrix is not jagged (i.e. rectangular or square)
		for i in range(len(m)):
			if(len(m[i]) != k):
				return False

		return True



