import scipy.io
import lasagne
import theano
import theano.tensor as T
import numpy as np
import time

import logging

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('experiment.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

TRAIN_NC = '../data/train.nc' # netcdf 
VAL_NC = '../data/val.nc' # netcdf
BATCH_SIZE = 50


def one_hot(labels, n_classes):
    one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
    one_hot[range(labels.shape[0]), labels] = True
    return one_hot


def load_netcdf(filename):
    with open(filename, 'r') as f:
        netcdf_data = scipy.io.netcdf_file(f).variables

    X = []
    y = []
    n = 0
    for length in netcdf_data['seqLengths'].data:
        X_n = netcdf_data['inputs'].data[n:n + length]
        X.append(X_n.astype(theano.config.floatX))
        y_n = one_hot(netcdf_data['targetClasses'].data[n:n + length],
                      netcdf_data['numTargetClasses'].data)
        y.append(y_n.astype(theano.config.floatX))
        n += length
    return X, y


def make_batches(X, length, batch_size=BATCH_SIZE):
    '''
    Convert a list of matrices into batches of uniform length

    :parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            matrix
    '''
    n_batches = len(X)//batch_size
    X_batch = np.zeros((n_batches, batch_size, length, X[0].shape[1]),
                       dtype=theano.config.floatX)
    X_mask = np.zeros((n_batches, batch_size, length), dtype=np.bool)
    for b in range(n_batches):
        for n in range(batch_size):
            X_m = X[b*batch_size + n]
            X_batch[b, n, :X_m.shape[0]] = X_m[:length]
            X_mask[b, n, :X_m.shape[0]] = 1
    return X_batch, X_mask


logger.info('Loading data...')
X_train, y_train = load_netcdf(TRAIN_NC)
X_train = X_train
y_train = y_train
X_val, y_val = load_netcdf(VAL_NC)
X_val = X_val
y_val = y_val

# Find the longest sequence
length = max(max([X.shape[0] for X in X_train]),
             max([X.shape[0] for X in X_val]))
# Convert to batches of time series of uniform length
X_train, _ = make_batches(X_train, length)
y_train, train_mask = make_batches(y_train, length)
X_val, _ = make_batches(X_val, length)
y_val, val_mask = make_batches(y_val, length)

n_epochs = 500
learning_rate = 1e-5
momentum = .9

l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, length, X_val.shape[-1]))
#l_noise = lasagne.layers.GaussianNoiseLayer(l_in, sigma=0.6)

l_forward_1 = lasagne.layers.LSTMLayer(l_in, num_units=156)
l_backward_1 = lasagne.layers.LSTMLayer(l_in, num_units=156,
                                        backwards=True)
l_recurrent_1 = lasagne.layers.ElemwiseSumLayer([l_forward_1, l_backward_1])

l_forward_2 = lasagne.layers.LSTMLayer(l_recurrent_1, num_units=300)
l_backward_2 = lasagne.layers.LSTMLayer(l_recurrent_1, num_units=300,
                                        backwards=True)
l_recurrent_2 = lasagne.layers.ElemwiseSumLayer([l_forward_2, l_backward_2])

l_forward_3 = lasagne.layers.LSTMLayer(l_recurrent_2, num_units=102)
l_backward_3 = lasagne.layers.LSTMLayer(l_recurrent_2, num_units=102,
                                        backwards=True)
l_recurrent_3 = lasagne.layers.ElemwiseSumLayer([l_forward_3, l_backward_3])

l_reshape = lasagne.layers.ReshapeLayer(l_recurrent_3,
                                       (BATCH_SIZE*length, 102))
nonlinearity = lasagne.nonlinearities.softmax
l_rec_out = lasagne.layers.DenseLayer(l_reshape, num_units=y_val.shape[-1],
                                      nonlinearity=nonlinearity)
l_out = lasagne.layers.ReshapeLayer(l_rec_out,
                                    (BATCH_SIZE, length, y_val.shape[-1]))


# Cost function is mean squared error
input = T.tensor3('input')
target_output = T.tensor3('target_output')
mask = T.matrix('mask')


def cost(output):
    return -T.sum(mask.dimshuffle(0, 1, 'x') * target_output*T.log(output))/T.sum(mask)

cost_train = cost(l_out.get_output(input, mask=mask, deterministic=False))
cost_eval = cost(l_out.get_output(input, mask=mask, deterministic=True))


# Use SGD for training
all_params = lasagne.layers.get_all_params(l_out)
logger.info('Computing updates...')
updates = lasagne.updates.momentum(cost_train, all_params,
                                   learning_rate, momentum)
logger.info('Compiling functions...')
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output, mask], cost_train,
                        updates=updates)
y_pred = theano.function([input, mask], l_out.get_output(input, mask=mask, deterministic=True))
compute_cost = theano.function([input, target_output, mask], cost_eval)

logger.info('Training...')
# Train the net
for epoch in range(n_epochs):
    start_time = time.time()
    batch_shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], False)
    for sequences, labels, sequence_mask in zip(X_train[batch_shuffle],
                                                y_train[batch_shuffle],
                                                train_mask[batch_shuffle]):
        sequence_shuffle = np.random.choice(sequences.shape[0],
                                            sequences.shape[0], False)
        train(sequences[sequence_shuffle], labels[sequence_shuffle],
              sequence_mask[sequence_shuffle])
    end_time = time.time()
    cost_val = sum([compute_cost(X_val_n, y_val_n, mask_n)
                    for X_val_n, y_val_n, mask_n,
                    in zip(X_val, y_val, val_mask)])
    y_val_pred = np.array([y_pred(X_val_n, mask_n) for X_val_n, mask_n in zip(X_val, val_mask)])
    y_val_labels = np.argmax(y_val*val_mask[:, :, :, np.newaxis], axis=-1).flatten()
    y_val_pred_labels = np.argmax(y_val_pred*val_mask[:, :, :, np.newaxis], axis=-1).flatten()
    n_time_steps = np.sum(val_mask)
    error = np.sum(y_val_labels != y_val_pred_labels)/float(n_time_steps)
    logger.info("Epoch {} took {}, cost = {}, error = {}".format(
        epoch, end_time - start_time, cost_val, error))

#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
#########################################################################
__author__ = "Thomas Rueckstiess, Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.structure.modules import LSTMLayer

batch=1 #number of samples per learning step
prnts=100 #number of learning steps after results are printed
epis=4000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts) #tool for printing and plotting

for runs in range(numbExp):
    # create environment
    env = CartPoleEnvironment()    
    # create task
    task = BalanceTask(env, 200, desiredValue=None)
    # create controller network
    net = buildNetwork(4, 2, 1, recurrent=True, hiddenclass=LSTMLayer, bias=False)
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(storeAllEvaluations = True))
    et.agent = agent
    # create the experiment
    experiment = EpisodicExperiment(task, agent)

    #Do the experiment
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()

