"""
This one still uses two lstm layer to predict R by minimizing R
"""

from lasagne.layers import shape
import numpy as np
import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import zeros
from random import choice
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask

np.set_printoptions()

# Start of sequence
START = 1

# End of Seuence
END = 20

# Points to be evaluated
POINTS = 1000

# Number of transmitted variables
N_TRANS = 5

# Input features
N_INPUT_FEATURES = 4

# Output Features
N_ACTIONS = 1

# Output Features
N_OUTPUT_FEATURES = 5

# Length of each input sequence of data
N_TIME_STEPS = 1  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 20

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 4e-1

# Number of iterations to train the net
N_ITERATIONS = 1000000

# Forget rate
FORGET_RATE = 0.9

# Number of reward output
N_REWARD = 1

GRADIENT_METHOD = 'sgd'



def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=theano.config.floatX).reshape(shape)


def one_iteration(task):
    """
    Give current value of weights, output all rewards
    :return:
    """
    observations = []
    rewards = []
    task.reset()
    while not task.isFinished():
        observation = task.getObservation()
        train_inputs = theano_form(observation, shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
        model_reward_result = action_prediction(train_inputs)
        task.performAction(model_reward_result*100)
        reward = task.getReward()
        observations.append(observation)
        rewards.append(reward)
    return observations, rewards
if __name__ == "__main__":
    import signal
    from einstein.data_structure import RingBuffer
    import lasagne as L


    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES))


    # Followed by LSTM Layer
    l_lstm_1 = lasagne.layers.LSTMLayer(input_layer=l_in,
                                           num_units=N_HIDDEN)

    l_lstm_reshape_1 = lasagne.layers.ReshapeLayer(input_layer=l_lstm_1,
                                            shape=(N_BATCH * N_TIME_STEPS, N_HIDDEN))

    # Followed by a Dense Layer to Produce Action
    l_action = lasagne.layers.DenseLayer(incoming=l_lstm_reshape_1,
                                                num_units=N_ACTIONS,
                                                nonlinearity=None,
                                                b=None)
    l_action_formed = lasagne.layers.ReshapeLayer(input_layer=l_action,
                                        shape=(N_BATCH, N_TIME_STEPS, N_ACTIONS))

    # Merge Action and Input Layer
    # This is three dimensional so merge over the most inside one (Which has axis as 2).
    l_merge = lasagne.layers.ConcatLayer([l_in, l_action_formed], axis=2)



    # Followed by LSTM Layer
    l_lstm_2 = lasagne.layers.LSTMLayer(input_layer=l_merge,
                                           num_units=N_HIDDEN)

    l_lstm_reshape_2 = lasagne.layers.ReshapeLayer(input_layer=l_lstm_2,
                                            shape=(N_BATCH * N_TIME_STEPS, N_HIDDEN))

    # Followed by a Dense Layer to Produce Output
    l_reward = lasagne.layers.DenseLayer(incoming=l_lstm_reshape_2,
                                                num_units=N_OUTPUT_FEATURES,
                                                nonlinearity=L.nonlinearities.identity)
    l_reward_formed = lasagne.layers.ReshapeLayer(input_layer=l_reward,
                                        shape=(N_BATCH, N_TIME_STEPS, N_OUTPUT_FEATURES))


    # Cost function is mean squared error
    input = T.tensor3('input')
    target_output = T.tensor3('target_output')


    # Cost = mean squared error, starting from delay point
    cost = T.mean((l_action_formed.get_output(input)[:, :, :]
                   - target_output[:, :, :])**2)


    # Use NAG for training
    all_params = lasagne.layers.get_all_params(l_action_formed)
    updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)

    # Theano functions for critic network,
    train = theano.function([input, target_output], cost, updates=updates)

    #y_pred_action = theano.function([input], l_action_formed.get_output(input))
    reward_prediction = theano.function([input], l_reward_formed.get_output(input))

    # Predict Action
    action_prediction = theano.function([input], l_action_formed.get_output(input))

    # Compute the cost
    compute_cost = theano.function([input, target_output], cost)

    # Training the network
    costs = np.zeros(N_ITERATIONS)




    # create environment
    env = CartPoleEnvironment()
    # create task
    task = BalanceTask(env, 200, desiredValue=None)

    action_prediction = theano.function([input], l_action_formed.get_output(input))


    all_params = lasagne.layers.get_all_params(l_reward_formed)

    baseline = None
    epsilon = 2 # initial number sigma
    best_reward = -1000

    for n in range(N_ITERATIONS):
        # Perform one episode with random initialization
        empirical_inputs, rewards = one_iteration(task)
        threshold = 1e1
        cost = 100
#        print input.shape
#        print theano_form(choice(empirical_inputs), shape=l_in.shape), zeros(l_in.shape)
        while cost > 1e1:
            cost = train(theano_form(choice(empirical_inputs), shape=l_in.shape), theano_form(zeros(l_in.shape), shape=l_in.shape))
        if not n%100:
            print sum(rewards)





"""
            train_inputs = all_data[:, 0:N_TIME_STEPS, 1::]
            model_reward_result = reward_prediction(train_inputs)
            model_action_result = action_prediction(train_inputs)

            # set desired output, the second number of result is reward
            train_outputs = all_data[
                            :,
                            1::, # extract reward from 1 to N_TIME_STEPS + 1,
                            :].copy().reshape([N_BATCH, N_TIME_STEPS, N_OUTPUT_FEATURES])  # Reward takes the first position
            train_outputs[:, :, 0] = 10

            costs[n] = train(train_inputs, train_outputs)
            if not n % 50:
                cost_val = compute_cost(train_inputs, train_outputs)
                #print "Iteration {} validation cost = {}".format(n, cost_val)
                print "reward predict: ", model_reward_result
                print

            # Extract the most recent action from all result.
            p = model_action_result[:, -1, 0]
            action = binomial(1, p, 1)
            serial.send("%d\0" % action)


"""
