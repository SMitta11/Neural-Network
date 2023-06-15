import pytest
import numpy as np

import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as kerass
from tensorflow.keras.datasets import mnist
from Mittal_04_01 import CNN

#testing the train method
def test_train():

    #seed the values
    tf.keras.utils.set_random_seed(5368)

    model = CNN()
    batch_size=10
    num_epochs=10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #100 samples of Xtrain
    X_train = X_train[0:100, :]
    X_train = X_train.astype('float32') / 255

    #20 samples of Xtrain
    X_test = X_test[0:20, :]
    X_test = X_test.astype('float32') / 255

    #100 samples of ytrain  
    y_train = y_train[0:100]

    #20 samples of y test
    y_test = y_test[0:20]

    model.add_input_layer(shape=(28, 28, 1))
    model.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name="conv1")
    model.append_maxpooling2d_layer(pool_size=5,padding="same",strides=1,name="maxpool1")
    model.append_conv2d_layer(num_of_filters=28, kernel_size=(4, 4), activation='relu', name="conv2")
    model.append_flatten_layer(name="flat1")
    model.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    model.set_optimizer(optimizer="SGD")
    model.set_loss_function(loss="hinge")
    model.set_metric(metric='accuracy')

    loss = model.train(X_train=X_train, y_train=y_train, batch_size=batch_size, num_epochs=num_epochs)
   
    assert loss[1] < loss[0]
    assert loss[2] < loss[1]
    assert loss[3] < loss[2]
    assert loss[4] < loss[3]
    assert np.allclose(loss[8],0.30400002002716064)



#testing the evaluate method
def test_evaluate():
    tf.keras.utils.set_random_seed(5368)
    model = CNN()
    batch_size = 10
    num_epochs = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
   

    X_train = X_train[0:100, :]
    X_train = X_train.astype('float32') / 255


    X_test = X_test[0:20, :]
    X_test = X_test.astype('float32') / 255


    y_train = y_train[0:100]
    y_test = y_test[0:20]

    model.add_input_layer(shape=(28, 28, 1))
    model.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name="conv1")
    model.append_maxpooling2d_layer(pool_size=5,padding="same",strides=1,name="maxpool1")
    model.append_conv2d_layer(num_of_filters=28, kernel_size=(4, 3), activation='relu', name="conv2")
    model.append_flatten_layer(name="flat1")
    model.append_dense_layer(num_nodes=10, activation="relu", name="dense1")
    model.set_optimizer(optimizer="SGD")
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_metric(metric='accuracy')
   

    loss, metric = model.evaluate(X=X_test, y=y_test)
    
    #print("loss, metric:",loss,metric)
    assert np.allclose(loss,2.316699266433716)
    assert np.allclose(metric,0.05000000074505806)