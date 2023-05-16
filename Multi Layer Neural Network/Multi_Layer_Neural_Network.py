import numpy as np
import copy
def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    return 1/(1+np.exp(-x))

#predict helper function
def predict(W,X):
    net = np.dot(W,X)
    #print(net,"net values")
    Y_Pred = sigmoid(net)
    return Y_Pred
    

#MSE calculation function
def MSE(Y_actual,Y_pred):
    return np.mean( ( Y_pred - Y_actual ) ** 2 )

#Partial derivative function
def PartialDerivate(f,X,h):
    return (f(X + h) - f(X - h)) / (2*h)

def setWeights(layers,seed,rows):
    weights = []
    layers_len = len(layers)
    for i in range(layers_len):
        #reseed value of weights for each layer
        weight_init = np.random.seed(seed)
        if i == 0:
            weight_init = np.random.randn(layers[i],rows+1)
        else:
            weight_init = np.random.randn(layers[i],layers[i-1]+1)
        weights.append(weight_init)
    return weights


def predictBatch(weights,n_layer,data):
    res = data
    for i in range(n_layer):
        res = np.insert(res, 0, 1, axis=0)
        w = weights[i]
        res = predict(w,res)
    return res

#Main function 
def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    
    err = []
    out = np.zeros((layers[-1],Y_test.shape[1]))

    num_layers = len(layers)
    rows_sample,column_sample = X_train.shape 
    weights = setWeights(layers,seed,rows_sample)
    final_weights = copy.deepcopy(weights)
    for epoch in range(epochs):
        #weights = setWeights(layers,seed,rows_sample)
        
        for i in range(num_layers):
            for j in range(weights[i].shape[0]):
                for k in range(weights[i].shape[1]):
                    weightValue = weights[i][j][k]
                    weights[i][j][k] = weightValue + h
                    #predict for current x train data w.r.t y train data
                    #call predict with +h data
                    predicted = predictBatch(weights,num_layers,X_train)
                    
                    mse_1 = MSE(Y_train,predicted)
                    weights[i][j][k] = weightValue - h
                    #call predict with -h data
                    predicted = predictBatch(weights,num_layers,X_train)
                    mse_2 = MSE(Y_train,predicted)
                    #predict for current x train data w.r.t y train data
                    weights[i][j][k] = weightValue
                    p_derivative = (mse_1 - mse_2)/(2 * h)
                    final_weights[i][j][k] = weights[i][j][k] - alpha * p_derivative
        #trainig ends
        weights = final_weights
        Y_pred_test = predictBatch(final_weights,num_layers,X_test)
        #print('Y_pred_test',Y_pred_test)
        
        mse = MSE(Y_test,Y_pred_test)
        err.append(mse)
        final_weights = copy.deepcopy(weights)
    out = predictBatch(final_weights,num_layers,X_test)
    return final_weights, err, out