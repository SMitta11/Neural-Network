

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred, n_classes=10):
    conf_matrix = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            conf_matrix[i, j] = np.sum(np.logical_and(y_true == i, y_pred == j))
    return conf_matrix


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    tf.keras.utils.set_random_seed(5368) # do not remove this line
   
    model = tf.keras.models.Sequential()
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers

     # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(layers.Conv2D(8,(3,3),activation='relu',padding='same',kernel_regularizer=regularizer.L2(0.0001)))
    # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(layers.Conv2D(16,(3,3),activation='relu',padding='same',kernel_regularizer=regularizer.L2(0.0001)))
    #Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
    # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_regularizer=regularizer.L2(0.0001)))
    # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',kernel_regularizer=regularizer.L2(0.0001)))
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))
    # - Flatten layer
    model.add(layers.Flatten())
    # - Dense layer with 512 units and ReLU activation
    model.add(layers.Dense(512, activation='relu',kernel_regularizer=regularizer.L2(0.0001)))
    # - Dense layer with 10 units with linear activation
    model.add(layers.Dense(10, activation='linear',kernel_regularizer=regularizer.L2(0.0001)))
    # - a softmax layer
    model.add(layers.Activation('softmax'))
   

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    history = model.fit(X_train, Y_train, epochs=epochs,batch_size=batch_size,validation_split=0.2)

    y_pred = model.predict(X_test)
  

    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred,axis=1)
    if Y_test.ndim == 2:
        Y_test = np.argmax(Y_test,axis=1)
    

    conf_matrix = confusion_matrix(Y_test,y_pred) 

    #save the model
    model.save('model.h5')

    #plot heat map of confusion matrix
    plt.matshow(conf_matrix)
    plt.xlabel('Y_pred')
    plt.ylabel('Y_true')
    plt.savefig('confusion_matrix.png')

    return model,history,conf_matrix,y_pred


    
    