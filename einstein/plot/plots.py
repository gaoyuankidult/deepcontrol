from numpy import array

import matplotlib.pyplot as p

from pickle import load



class Plot(object):
    def __init__(self, name):
        self.p = p
        self.name = name
        pass
    

class BoxPlot(Plot):
    def __init__(self, name = "boxplot"):
        assert  isinstance(name, basestring)
        super(BoxPlot, self).__init__(name)

    def plot(self, data):
        self.p.boxplot(data)
        self.p.savefig(self.name + ".png")

    def plot(self, filename):
        self.f = open(filename, "rb")
        data = array(load(open(filename, "rb")))
        self.plot(data=data)

        



