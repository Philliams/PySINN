from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Metric(Layer):
    def __init__(self, initiation_matrix, **kwargs):

        # my variables
        self.input_dim = len(initiation_matrix[0])
        self.nodes     = len(initiation_matrix)

        self.B = K.variable(initiation_matrix)
        self.P = K.variable(2)

        # keras variables
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []

        # keras constructor call
        super(Metric, self).__init__(**kwargs)

    def build(self,input_shape):

        # initiate weights
        self.W = K.variable(np.full((self.nodes, self.input_dim), 1,dtype='float64'))
        self.A = K.variable(np.full((self.nodes, self.input_dim), 1,dtype='float64'))

        # tell keras which ones are trained
        self.trainable_weights = [self.W,self.B,self.A]

        # keras constructor call
        super(Metric, self).build(input_shape)

    def call(self, x, mask=None):
        # element wise PaReLU on a single input_vector
        batch_size = K.shape(x)[0]

        x = K.expand_dims(x, dim=1)
        x = K.repeat_elements(x, self.nodes, 1)

        W = K.expand_dims(self.W, dim=0)
        W = K.repeat_elements(W,batch_size,0)

        A = K.expand_dims(self.A, dim=0)
        A = K.repeat_elements(A,batch_size,0)

        B = K.expand_dims(self.B, dim=0)
        B = K.repeat_elements(B,batch_size,0)

        result = 1 / ( K.sum(K.abs(A * (W * x - B)), axis=2) + 1e-6)
        denominator = K.max(result, axis = 1)
        denominator = K.expand_dims(denominator,dim=1)
        denominator = K.repeat_elements(denominator, self.nodes, axis=1)

        normalized = result / denominator
        
        return K.pow(normalized,self.P)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.nodes)

class Shepard(Layer):
    def __init__(self, initiation_matrix, **kwargs):

        # my variables
        self.nodes      = len(initiation_matrix)
        self.output_dim = len(initiation_matrix[0])

        # first nested array is nodes
        # second is dimensions
        self.W = K.variable(np.transpose(initiation_matrix))

        # keras variables
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = [self.W]

        # keras constructor call
        super(Shepard, self).__init__(**kwargs)

    def build(self,input_shape):
        # tell keras which ones are trained
        self.trainable_weights = []
        # keras constructor call
        super(Shepard, self).build(input_shape)

    def call(self, x, mask=None):
        # element wise PaReLU on a single input_vector
        batch_size = K.shape(x)[0]
        input_size = K.shape(x)[1]

        y = K.sum(x,axis=1)
        y = K.transpose(y)
        y = K.expand_dims(y,dim=1)
        y = K.repeat_elements(y,input_size,1)

        x = x / ( y + 1e-6)

        x = K.expand_dims(x, dim=1)
        x = K.repeat_elements(x, self.output_dim, 1)
        
        W = K.expand_dims(self.W, dim=0)
        W = K.repeat_elements(W,batch_size,0)

        return K.sum(W*x,axis=2)

    def get_output_shape_for(self, input_shape):
        return (None,self.output_dim)



