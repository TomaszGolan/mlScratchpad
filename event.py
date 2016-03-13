import numpy
import matplotlib.pyplot as plt

rng = numpy.random # random number generator

### DETECTOR SETTINGS ###

dim = [100,50]      # dimension [width, height]
N = dim[0] * dim[1] # number of blocks in the detector

### EVENT SETTINGS ###

nTracks = 5 # maximum no. of tracks

class Event:
    # initialize with random vertex and generate tracks
    def __init__ (self):
        # vertex = [x,y]
        self.vertex =  dim * rng.sample(2)
        # each event has random number of tracks
        self.tracks = [self.genTrack() for i in range(rng.randint(2,nTracks))]
        # data represents as array of 0 (no track) and 1 (track) points in the detector
        self.data = [0] * N
        # fill data with tracks
        for t in self.tracks: self.genData(t[0], t[1])
        self.data2d = numpy.array(self.data, dtype='float32').reshape(dim[1], dim[0])
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

class EventDisplay:
    def __init__(self, e):
        # axis ranges
        plt.xlim([0,dim[0]])
        plt.ylim([0,dim[1]])
        # ticks
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        # data
        X = []
        Y = []
        for i in range(dim[0]):
            for j in range(dim[1]):
                if e.data[i + dim[0] * j]:
                    X.append(i)
                    Y.append(j)
        plt.scatter(X, Y, color = 'k', marker = 's')
        # true tracks
        x = numpy.arange(e.vertex[0], dim[0], 1)
        for t in e.tracks:
            plt.plot(x, t[0] * x + t[1], 'b')
        # vetex
        plt.scatter(e.vertex[0], e.vertex[1], color = 'r', marker = 's')
        plt.show()
