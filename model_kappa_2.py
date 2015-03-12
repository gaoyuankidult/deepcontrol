"""
This is exact copy of model_eta, and now I modify it
"""

import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import ones
from numpy.random import uniform
from numpy import exp, log, sign

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

def sample_parameter(sigmas):
    """
    sigma_list contains sigma for each parameters
    """
    def abig(a):
        c1 = - 0.06655
        c2 = - 0.9706
        return exp(c1 * (abs(a)**3 - abs(a)) / log(abs(a)) + c2 * abs(a))
    def asmall(a):
        c3 = 0.124
        return exp(a)/(1.0 - a ** 3) ** (c3 *a)

    # normal sampling
    epsilon = np.random.normal(0., sigmas)
    theta = 0.67449 * sigmas
    mirror_sigma_samples = np.random.normal(0., theta)
    a = (theta -abs(epsilon)) / theta
    f_maps = [abig if x > 0 else asmall for x in a ]
    epsilon_star = sign(epsilon) * theta * array([v(x)  for v, x in zip(f_maps, a)])
    return epsilon, epsilon_star



if __name__ == "__main__":

    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer

    # This is an actor model
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
    for time in xrange(50):
        records.append([])
        _all_params = lasagne.layers.get_all_params(l_action_formed)
        _all_params[0].set_value(theano_form(uniform(-0.1, 0.1, 4), shape=(4,1)))


        baseline = None
        num_parameters = 4 # five parameters
        init_sigma = 3 # initial number sigma
        sigmas = ones(num_parameters) * init_sigma
        best_reward = -1000
        current = all_params[0].get_value()[:, 0]
        arg_reward = []

        for n in xrange(1500):

             # current parameters
            epsilon, epsilon_star = sample_parameter(sigmas=sigmas)
            reward1 = one_iteration(task=task, all_params=current + epsilon)
            if reward1 > best_reward:
                best_reward = reward1
            reward2 = one_iteration(task= task, all_params=current - epsilon)
            if reward2 > best_reward:
                best_reward = reward2

            reward3 = one_iteration(task=task, all_params= current + epsilon_star)
            if reward3 > best_reward:
                best_reward = reward3

            reward4 = one_iteration(task=task, all_params= current - epsilon_star)
            if reward4 > best_reward:
                best_reward = reward4


            mreward1 = (reward1 + reward2) / 2.
            mreward2 = (reward3 + reward4) / 2.

            if baseline is None:
                # first learning step
                baseline = mreward1
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
                    fakt2=(mreward1-baseline)/(best_reward-baseline)
                else:
                    fakt2 = 0.0
            #update baseline
            baseline = 0.9 * baseline + 0.1 * mreward1
            # update parameters and sigmas
            current = current + LEARNING_RATE * fakt * epsilon

            if mreward1 - mreward2 >= 0.: #for sigma adaption alg. follows only positive gradients
                #apply sigma update locally
                #sigma_list = sigma_list + LEARNING_RATE * fakt2 * () / sigma_list
                sigmas = sigmas + 0.0009 * (epsilon * epsilon - sigmas * sigmas)/sigmas/2.0*\
                                          (mreward1 - mreward2)
            else:
                sigmas = sigmas + 0.0009 * (epsilon_star * epsilon_star - sigmas * sigmas)/sigmas/2.0*\
                                          (mreward2-mreward1)

            arg_reward.append((mreward1 + mreward2)/2)
            if not n%100:
                temp_arg = sum(arg_reward)/len(arg_reward)
                records[time].append(temp_arg)
                print "best reward", best_reward, "average reward", temp_arg
                arg_reward = []
    print records
    pickle.dump(records, open("records_super_sys.p", "wb"))
