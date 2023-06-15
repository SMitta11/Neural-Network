
import numpy as np
import tensorflow as tf

#Initialize weigths
def weights_init(layers,seed,column):
    weights=[]
    n_layers = len(layers)
    for i in range(n_layers):
        w = np.random.seed(seed)
        if i == 0:
            w = np.random.randn(column+1,layers[i])
        else:
            w = np.random.randn(layers[i-1]+1,layers[i])
        weights.append(np.float32(w))
    return weights
        
#Split data in training and validation
def split_data(X_train, Y_train, validation_split):
    start = int(validation_split[0] * X_train.shape[0])
    end = int(validation_split[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]

# #Generate batches for training data
def generate_batches(X, y,batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]  
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

def getActivations(activation,pred):
    if activation.lower() == "sigmoid":
        return tf.math.sigmoid(pred) 
    elif activation.lower() == "relu":
        return tf.nn.relu(pred)
    elif activation.lower() == "linear":
        return pred
    
def predict(X,weights,activations):
    prev_layer = X
    for i in range(len(weights)):
        ones = tf.ones(shape=[tf.shape(prev_layer)[0], 1], dtype=prev_layer.dtype)
        prev_layer = tf.concat([ones, prev_layer], axis=1)
        prev_layer = tf.cast(prev_layer, dtype=tf.float32)
        temp = tf.matmul(prev_layer,weights[i])
        prev_layer = getActivations(activations[i],temp)
    return prev_layer

def calculateLoss(Y_actual, Y_pred, loss):
    if loss.lower() == "mse" :
        return tf.reduce_mean(tf.square(Y_actual - Y_pred ))
    elif loss.lower() == "svm":
        return  tf.reduce_mean(tf.square(Y_actual - Y_pred ))

    elif loss.lower() == "cross_entropy":
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_actual, logits = Y_pred))
    
def getTensor(data):
    for i in range(len(data)):
        data[i] = tf.Variable(data[i],dtype=tf.float32)
    return data

def getNumpy(data):
    for i in range(len(data)):
        data[i] = data[i].numpy()
    return data

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):
    err = []
    out = []
    column = X_train.shape[1]
    if not weights:
        weights = weights_init(layers,seed,column)
    X_train1, Y_train1, X_val, Y_val = split_data(X_train,Y_train,validation_split)
    for epoch in range( epochs):
        for X_batch, Y_batch in generate_batches( X_train1, Y_train1, batch_size ):
            with tf.GradientTape(persistent=True) as tape:
                tfweight = getTensor(weights) 
                tape.watch(tfweight)
                Y_pred = predict( X_batch, weights, activations )
                loss_pred = calculateLoss( Y_batch, Y_pred, loss )
            gradient = tape.gradient( loss_pred, tfweight)
            for k in range(len(tfweight)):
                tfweight[k].assign_sub(alpha * gradient[k])
        weights = getNumpy(tfweight)
        Y_pred_val = predict( X_val, weights, activations)
        loss_val = calculateLoss( Y_val, Y_pred_val, loss)
        err.append(loss_val)
    out = predict( X_val, weights, activations )
    return weights,err,out
