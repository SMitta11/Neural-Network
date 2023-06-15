# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import keras
# import tensorflow.keras as keras

class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network

        """
        
        self.model = keras.models.Sequential()
    


    def add_input_layer(self, shape=(2,),name="" ):
        """
         This method adds an input layer to the neural network. If an input layer exist, then this method
         should replace it with the new input layer.
         Input layer is considered layer number 0, and it does not have any weights. Its purpose is to determine
         the shape of the input tensor and distribute it to the next layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        self.model.add(keras.layers.InputLayer(input_shape=shape,name = name))


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This method adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        self.model.add(keras.layers.Dense(num_nodes,activation=activation,name=name,trainable=trainable))

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This method adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        return self.model.add(keras.layers.Conv2D(filters=num_of_filters,kernel_size=kernel_size,strides=strides,padding=padding,activation=activation,trainable=trainable,name=name))
    
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This method adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        return self.model.add(keras.layers.MaxPool2D(pool_size=pool_size,padding=padding,strides=strides,name=name))

    def append_flatten_layer(self,name=""):
        """
         This method adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        return self.model.add(keras.layers.Flatten(name=name))
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This method sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        if layer_numbers:
            if type(layer_numbers) is list:
                for i in layer_numbers:
                    self.model.layers[i].trainable = trainable_flag
            else:
                self.model.layers[layer_numbers].trainable = trainable_flag
        else:
            if type(layer_names) is list:
                for i in layer_names:
                    self.model.get_layer(name = i).trainable =  trainable_flag
            else:
                self.model.get_layer(name = layer_names).trainable = trainable_flag

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This method should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """
     
        if layer_number is None:
            temp = self.model.get_layer(name = layer_name)
            if len(temp.get_weights()) > 0:
                weight_without_bias = temp.get_weights()[0]
                return weight_without_bias 
        elif layer_number > 0:
            weight_without_bias = self.model.layers[layer_number - 1].get_weights()[0]
            if len(weight_without_bias) > 0:
                return weight_without_bias
        elif layer_number < 0:
                return self.model.layers[layer_number].get_weights()[0]
    

    def get_biases(self,layer_number=None,layer_name=""):
        """
        This method should return the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        if layer_number is None:
            temp = self.model.get_layer(name = layer_name)
            if len(temp.get_weights()) > 0:
                weight_without_bias = temp.get_weights()[1]
                return weight_without_bias 
        elif layer_number > 0:
            weight_without_bias = self.model.layers[layer_number - 1].get_weights()[1]
            if len(weight_without_bias) > 0:
                return weight_without_bias
        elif layer_number < 0:
                return self.model.layers[layer_number].get_weights()[1]

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This method sets the weight matrix for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """


        if layer_number is None:
            temp = self.model.get_layer(name = layer_name)
            if len(temp.get_weights()) > 0:
                bias =  temp.get_weights()[1]
                temp.set_weights([weights,bias])
        elif layer_number:
            temp = self.model.layers[layer_number - 1]
            if len(temp.get_weights()) > 0:
                bias =  temp.get_weights()[1]
                temp.set_weights([weights,bias])


       


    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This method sets the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        if layer_number is None:
            temp = self.model.get_layer(name = layer_name)
            if len(temp.get_weights()) > 0:
                weights =  temp.get_weights()[0]
                temp.set_weights([weights,biases])
        elif layer_number:
            temp = self.model.layers[layer_number - 1]
            if len(temp.get_weights()) > 0:
                weights =  temp.get_weights()[0]
                temp.set_weights([weights,biases])

    def remove_last_layer(self):
        #get last layer from current model
        last_layer = self.model.get_layer(index=-1)

        #save existing model to model
        model = self.model

        #create new model with same config as existing one
        self.model = keras.Sequential.from_config(model.get_config())
        #pop the last layer from this new model
        self.model.pop()
        return last_layer




    def load_a_model(self,model_name="",model_file_name=""):
        """
        This method loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """

        if model_name:
            if model_name == "VGG16":
                model = keras.applications.VGG16()
            elif model_name == "VGG19":
                model = keras.applications.VGG19()
            self.model = keras.Sequential.from_config(model.get_config()) #creates new sequential model with same conig as model object
        else:
            self.model = keras.models.load_model(model_file_name)

        return self.model

    def save_model(self,model_file_name=""):
        """
        This method saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        self.model.save(model_file_name)
        return self.model

    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This method sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """

        if loss.lower() == "SparseCategoricalCrossentropy".lower():
            self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif loss.lower() == "MeanSquaredError".lower():
            self.loss = keras.losses.MeanSquaredError()
        elif loss.lower() == "hinge".lower():
            self.loss = keras.losses.Hinge()

    def set_metric(self,metric):
        """
        This method sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        if metric == "accuracy":
            self.metric = ['accuracy']
        elif metric == "mse":
            self.metric = ['mse']

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This method sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
       
        if optimizer.lower() == "SGD".lower():
            self.optimizer = keras.optimizers.SGD(learning_rate = learning_rate , momentum = momentum)
        elif optimizer.lower() == "RMSprop".lower():
            self.optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate , momentum = momentum)
        elif optimizer.lower() == "Adagrad".lower():
       
            self.optimizer = keras.optimizers.Adagrad(learning_rate = learning_rate )

    def predict(self, X):
        """
        Given array of inputs, this method calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        return self.model.predict(X)

    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this method returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        self.model.compile (optimizer = self.optimizer , loss = self.loss , metrics = self.metric)
        return self.model.evaluate(x = X , y = y)

    def train(self, X_train, y_train, batch_size, num_epochs):

        """
         Given a batch of data, and the necessary hyperparameters,
         this method trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        self.model.compile (optimizer = self.optimizer , loss = self.loss , metrics = self.metric)
        data = self.model.fit (x = X_train , y = y_train ,batch_size = batch_size , epochs = num_epochs )
        return data.history['loss']



