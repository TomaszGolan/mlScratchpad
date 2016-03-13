### IMPORTS ###

from __future__ import print_function

import sys
import os
import time
import math

import numpy as np
import theano
import theano.tensor as T

import lasagne

import event

### SETTINGS ###

f = open('/home/goran/theano/toyDetectorConv.log', 'w')

nTrainSamples = 10000
nEpochs = 50
nTestSamples = 1000

# training params

alpha = 0.01
momentum = 0.9

### FUNCTIONS ###

def genData(N):
    """Generate N events"""
    X = []
    Y = []
    for i in range(N):
        e = event.Event()
        X.append(e.data2d.reshape((1, 1, event.dim[1], event.dim[0])))
        Y.append(e.vertex[0] / event.dim[0])
    return X,  np.array(Y, dtype='float32')

### CONVOLUTIONAL NEURAL NETWORK ###

def build_cnn(input_var=None):
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, event.dim[1], event.dim[0]),
                                        input_var=input_var)
    # Convolution layer
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Convolution layer
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Fully-connected layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # Output layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

### MAIN FUNCTION ###

def main():
    x = T.tensor4('inputs') # tensor4 required by theano conv2d
    y = T.scalar('targets') # x-position

    network = build_cnn(x) # build convolutional network

    prediction = lasagne.layers.get_output(network) # x-pos prediction
    # cost function (for regression!)
    cost = lasagne.objectives.squared_error(prediction, y).mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            cost, params, learning_rate=alpha, momentum=momentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    # TODO what does it mean...
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_cost = lasagne.objectives.squared_error(test_prediction, y).mean()

    train = theano.function([x, y], cost, updates=updates)

    val = theano.function([x, y], [test_cost, test_prediction])

    X, Y = genData(nTrainSamples)        # generate training samples
    testX, testY = genData(nTestSamples) # generate testing samples

    for epoch in range(nEpochs):
        # loop over epochs
        start_time = time.time()
        # loop over training samples
        for i in range(nTrainSamples): train(X[i], Y[i])

        print("Epoch {} of {} took {:.3f}s".format(
               epoch + 1, nEpochs, time.time() - start_time), file=f)

        # test epoch

        good = [0] * 5 # good defined as within 5 planes
        bad = 0        # otherwise is not reconstructed

        for i in range(nTestSamples):
            trueV = int(testY[i] * event.dim[0])
            predV = int(val(testX[i], testY[i])[1] * event.dim[0])
            diff = int(math.fabs(trueV - predV))
            if diff < 5: good[diff] += 1
            else: bad += 1

        print ('\nReconstruted events: %.2f%%'
               % (100.0 * sum(good) / nTestSamples), file=f)

        for i in range(5):
            print ("\t within %d planes: %f%%"
                   % (i + 1, 100.0 * good[i] / nTestSamples), file=f)

        print ('', file=f)

        f.flush() # refresh file after each epoch

if __name__ == '__main__':
    main()
