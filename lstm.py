import numpy as np
import theano
import theano.tensor as T
import lasagne

from numpy import array
from numpy import linspace
from numpy import sin as np_sin
from numpy.random import choice
from serial.socket import SocketServer

# Start of sequence
START = 1

# End of Seuence
END = 20

# Points to be evaluated
POINTS = 1000

# Sequence length
LENGTH = 1
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 2000
# Number of training sequences in each batch
N_BATCH = 10
# Delay used to generate artificial training data
DELAY = 1
# SGD learning rate
LEARNING_RATE = 1e-4
# Number of iterations to train the net
N_ITERATIONS = 1000

GRADIENT_METHOD = 'sgd'

class SeqGenerator():

    def __init__(self, fn_type='linear'):
        self.type = fn_type

    def gen_data(self, start, end, n_points, n_batch, length, cls='sgd'):
        """
        generate data according to the function and class type.

        for sdg class, the function randomly sample points from the list
        for sequencial class, the function sequencially put the points into the lists.
        """

        if self.type == 'linear':
            x = array([linspace(start, end, n_points), linspace(start, end, n_points)]).T
        if self.type == 'sin':
            x = array([linspace(start, end, n_points), np_sin(linspace(start, end, n_points))]).T


        if cls == 'sgd':
            x1_indexes = choice(len(x)-DELAY, [n_batch, length]) # enable 2000 - 1 = 1999 combinations
            x1 = x[x1_indexes]
            x2 = x[x1_indexes + DELAY]

        return x1.astype(theano.config.floatX), x2.astype(theano.config.floatX)

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        x1, x2 = seq_gen.gen_data(START, END, POINTS, N_BATCH, LENGTH, cls=GRADIENT_METHOD)
        print "given x:"
        print x1
        print "now predicting...y"
        print y_pred(x1)
        sys.exit(0)

if __name__ == "__main__":
    import signal
    import sys

    seq_gen = SeqGenerator(fn_type='sin')

    # Generate a "validation" sequence whose cost we will periodically compute
    x_val, y_val = seq_gen.gen_data(START, END, POINTS, N_BATCH, LENGTH, cls=GRADIENT_METHOD)

    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, x_val.shape[-1]))

    l_recurrent = lasagne.layers.LSTMLayer(l_in, N_HIDDEN)
    l_reshape = lasagne.layers.ReshapeLayer(l_recurrent,
                                            (N_BATCH*LENGTH, N_HIDDEN))

    l_recurrent_out = lasagne.layers.DenseLayer(l_reshape,
                                                num_units=y_val.shape[-1],
                                                nonlinearity=None)
    l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                        (N_BATCH, LENGTH, y_val.shape[-1]))

    # Cost function is mean squared error
    input = T.tensor3('input')
    target_output = T.tensor3('target_output')
    # Cost = mean squared error, starting from delay point
    cost = T.mean((l_out.get_output(input)[:, :, :]
                   - target_output[:, :, :])**2)
    # Use NAG for training
    all_params = lasagne.layers.get_all_params(l_out)

    updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
    # Theano functions for training, getting output, and computing cost
    train = theano.function([input, target_output], cost, updates=updates)
    y_pred = theano.function([input], l_out.get_output(input))
    compute_cost = theano.function([input, target_output], cost)

    # Train the net
    costs = np.zeros(N_ITERATIONS)

    # Catch the Ctrl+C signal, if there is any and give it to signal handler.
    signal.signal(signal.SIGINT, signal_handler)
    #################################################

    serial = SocketServer()

    for n in range(N_ITERATIONS):

        x, x_dot, theta, theta_dot = signal.split()
        x1, x2 = seq_gen.gen_data(START, END, POINTS, N_BATCH, LENGTH, cls=GRADIENT_METHOD)
        costs[n] = train(x1, x2)
        if not n % 10:
            cost_val = compute_cost(x_val, y_val)
            print "Iteration {} validation cost = {}".format(n, cost_val)

    print "Training finished..."
    from numpy import array
    x1, x2 = seq_gen.gen_data(START, END, POINTS, N_BATCH, LENGTH, cls=GRADIENT_METHOD)
    print "given x:"
    print x1
    print "now predicting...y"
    print y_pred(x1)


    import matplotlib.pyplot as plt
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    plt.savefig('img.png')