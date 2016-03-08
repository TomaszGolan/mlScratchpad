### IMPORTS ###

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy
import math

rng = numpy.random # random number generator

### OUTPUT FILE ###

f = open('/home/goran/theano/toyDetector.log', 'w')

### NET SETTINGS ###

nTrainSamples = 100000 # number of learning samples
nTrainSteps = 400      # number of training steps
alpha = 0.1            # learning rate
nTestSamples = 1000    # number of testing samples

h1 = 15 # no. of neurons in first hidden layer
h2 = 10 # no. of neurons in second hidden layer

### DETECTOR SETTINGS ###

dim = [100,50]      # dimension [width, height]
N = dim[0] * dim[1] # number of blocks in the detector

### EVENT ###

nTracks = 5 # maximum no. of tracks

class Event:
    # initialize with random vertex and generate tracks
    def __init__ (self):
        # vertex = [x,y]
        self.vertex =  dim * rng.sample(2)
        self.vertex[1] = dim[1] / 2 # in this version all vertices at y-center
        # each event has random number of tracks
        self.tracks = [self.genTrack() for i in range(rng.randint(2,nTracks))]
        # data represents as array of 0 (no track) and 1 (track) points in the detector
        self.data = [0] * N
        # fill data with tracks
        for t in self.tracks: self.genData(t[0], t[1])
    # generate track (random line coming from vertex)
    def genTrack (self):
        # y = ax + b
        a = 2.0 * rng.sample() - 1.0
        b = self.vertex[1] - a * self.vertex[0]
        return [a,b]
    # generate data from tracks
    def genData (self, a, b):
        # track starts in the vertex
        start = int(self.vertex[0])
        # track length is random (but not smaller than 5)
        end = start + int((dim[0] - start) * rng.sample())
        # "convert" track to detector points
        for x in range(start, end):
            y = int(a * x + b)  # round y
            if y < 0 or y >= dim[1]: break
            self.data[x + y * dim[0]] = 1


### GENERATE TRAINING SAMPLES ###

X = []
Y = []

for i in range(nTrainSamples):
    e = Event()
    X.append(e.data)
    Y.append(e.vertex[0] / dim[0])

Y = numpy.array(Y, dtype='float32')

###### SYMBOLIC VARIABLES ###

x = T.vector('x') # input
y = T.scalar('y') # expected value

w1 = theano.shared(rng.rand(N + 1, h1), name = 'w1')
w2 = theano.shared(rng.rand(h1 + 1, h2), name = 'w2')
w3 = theano.shared(rng.rand(h2 + 1), name = 'w3')

### EXPRESSION GRAPH ###

def hlayer (x, w): # relu for hidden layers
    b = numpy.array([1])      # bias term
    xb = T.concatenate([x,b]) # input x with bias added
    return nnet.sigmoid(T.dot(w.T,xb))

def layer (x, w): # sigmoid for output layer
    b = numpy.array([1])      # bias term
    xb = T.concatenate([x,b]) # input x with bias added
    return nnet.sigmoid(T.dot(w.T,xb))

hiddenLayer  = hlayer (x, w1)                 # hidden layer 1
hiddenLayer2 = hlayer (hiddenLayer, w2)       # hidden layer 2
outputLayer  = T.sum(layer(hiddenLayer2, w3)) # output layer

cost = (outputLayer - y)**2                   # cost function

def gradient (c, w):                 # cost function, weights
    return w - alpha * T.grad (c, w) # update weights

### COMPILE ###

train = theano.function(inputs = [x,y],
                        outputs = cost,
                        updates = [(w1, gradient(cost, w1)),
                                   (w2, gradient(cost, w2)),
                                   (w3, gradient(cost, w3))])

predict = theano.function(inputs=[x], outputs=outputLayer)

### TEST ###

testSet = [Event() for i in range(nTestSamples)]

def test():
    good = [0] * 5
    bad = 0

    for e in testSet:
        diff = int(math.fabs(e.vertex[0] - predict(e.data) * dim[0]))
        if diff < 5: good[diff] += 1
        else: bad += 1

    print >> f, '\nReconstruted events: %.2f%%' % (100.0 * sum(good) / nTestSamples)

    for i in range(5):
        print >> f, "\t within %d planes: %f%%" % (i + 1, 100.0 * good[i] / nTestSamples)
    print >> f, ''

### TRAIN ###

for i in range (nTrainSteps):
    for j in range(nTrainSamples):
        c = train(X[j], Y[j])
    print >> f, 'Epoch: %d, cost = %f' % (i + 1, c)
    if (i + 1) % 10 == 0: test()
    f.flush()

f.close()
