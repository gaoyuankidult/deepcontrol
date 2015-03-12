"""
This is exact copy of model_eta
"""

import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import ones
from numpy.random import uniform

import pickle

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
N_OUTPUT_FEATURES = 4

# Length of each input sequence of data
N_TIME_STEPS = 1  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 1e-1

# Number of iterations to train the net
N_ITERATIONS = 1000000


def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=theano.config.floatX).reshape(shape)


def one_iteration(task, all_params):
    """
    Give current value of weights, output all rewards
    :return:
    """
    rewards = []
    _all_params = lasagne.layers.get_all_params(l_action_formed)
    _all_params[0].set_value(theano_form(all_params, shape=(4, 1)))
    task.reset()
    while not task.isFinished():
        train_inputs = theano_form(task.getObservation(), shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
        model_reward_result = action_prediction(train_inputs)
        task.performAction(model_reward_result)
        rewards.append(task.getReward())
    return sum(rewards)

def sample_parameter(sigma_list):
    """
    sigma_list contains sigma for each parameters
    """
    return np.random.normal(0., sigma_list)



if __name__ == "__main__":

    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES))

    # Followed by a Dense Layer to Produce Action
    l_action = lasagne.layers.DenseLayer(incoming=l_in,
                                         W=lasagne.init.Uniform([-0.1, 0.1]),
                                         num_units=N_ACTIONS,
                                         nonlinearity=None,
                                         b=None)

    l_action_formed = lasagne.layers.ReshapeLayer(input_layer=l_action,
                                        shape=(N_BATCH, N_TIME_STEPS, N_ACTIONS))


    # Cost function is mean squared error
    input = T.tensor3('input')
    target_output = T.tensor3('target_output')

    # create environment
    env = CartPoleEnvironment()
    # create task
    task = BalanceTask(env, 200, desiredValue=None)

    #
    action_prediction = theano.function([input], l_action_formed.get_output(input))


    all_params = lasagne.layers.get_all_params(l_action_formed)

    records = []
    total_counts = []
    for time in xrange(50):
        records.append([])
        _all_params = lasagne.layers.get_all_params(l_action_formed)
        _all_params[0].set_value(theano_form(uniform(-0.1, 0.1, 4), shape=(4,1)))


        baseline = None
        num_parameters = 4 # five parameters
        epsilon = 3 # initial number sigma
        sigma_list = ones(num_parameters) * epsilon
        deltas = sample_parameter(sigma_list=sigma_list)
        best_reward = -1000
        current = all_params[0].get_value()[:, 0]
        arg_reward = []

        total_count = 0
        for n in xrange(1500):
            # current parameters
            deltas = sample_parameter(sigma_list=sigma_list)
            reward1 = one_iteration(task=task, all_params=current + deltas)
            total_count += 1
            if reward1 > best_reward:
                best_reward = reward1
            reward2 = one_iteration(task= task, all_params=current - deltas)
            total_count += 1
            if reward2 > best_reward:
                best_reward = reward2
            mreward = (reward1 + reward2) / 2.

            if baseline is None:
                # first learning step
                baseline = mreward
                fakt = 0.
                fakt2 = 0.
            else:
                #calc the gradients
                if reward1 != reward2:
                    #gradient estimate alla SPSA but with likelihood gradient and normalization
                    fakt = (reward1 - reward2) / (2. * best_reward - reward1 - reward2)
                else:
                    fakt=0.
                #normalized sigma gradient with moving average baseline
                norm = (best_reward - baseline)
                if norm != 0.0:
                    fakt2=(mreward-baseline)/(best_reward-baseline)
                else:
                    fakt2 = 0.0
            #update baseline
            baseline = 0.9 * baseline + 0.1 * mreward
            # update parameters and sigmas
            current = current + LEARNING_RATE * fakt * deltas

            if fakt2 > 0.: #for sigma adaption alg. follows only positive gradients
                #apply sigma update locally
                sigma_list = sigma_list + LEARNING_RATE * fakt2 * (deltas * deltas - sigma_list * sigma_list) / sigma_list


            # Test set
            deltas = sample_parameter(sigma_list=sigma_list)
            test_reward1 = one_iteration(task=task, all_params=current + deltas)
            test_reward2 = one_iteration(task= task, all_params=current - deltas)
            test_mreward = (test_reward1 + test_reward2)/ 2.0


            arg_reward.append(test_mreward)
            if not n%100:
                temp_arg = sum(arg_reward)/len(arg_reward)
                records[time].append(temp_arg)
                print "best reward", best_reward, "average reward", temp_arg
                arg_reward = []
                print total_count
            total_counts.append(total_count)
    print records
    pickle.dump(records, open("records_kappa_eval.p", "wb"))
    pickle.dump(total_counts, open("total_counts_kappa_eval.p", "wb"))
