"""
This is lambda model of system. It uses LSTM layer to predict rewards to reduce number real simulation.

This is second version of model lambda. It tries to predict next state as well in order to calculate reward by simulation.
"""

import theano as T
import theano.tensor as TT
import lasagne

import numpy as np
from numpy import array
from numpy import ones
from numpy.random import uniform
from numpy import exp, log, sign, mean
from numpy import concatenate

from scipy import random

import pickle

from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.environments import EpisodicTask

from einstein.data_structure import RingBuffer, theano_form

class SimBalanceTask(EpisodicTask):

    randomInitialization = True
    def __init__(self, prediction, maxsteps):
        super(SimBalanceTask, self).__init__(None)
        self.prediction = prediction
        self.sensors_sequence = RingBuffer(N_CTIME_STEPS, ivalue=[0.0] * 4)
        self.actions_sequence = RingBuffer(N_CTIME_STEPS, ivalue=[0.0])
        self.sensors = self.sensors_sequence.data[-1]
        self.t = 0
        self.N = maxsteps

    def performAction(self, action):
        self.t += 1
        self.actions_sequence.append(action[0][0])
        predict_input = concatenate([theano_form(self.actions_sequence.data, shape=(N_CBATCH, N_CTIME_STEPS, 1)),
                                     theano_form(self.sensors_sequence.data, shape=(N_CBATCH, N_CTIME_STEPS, 4))], axis=2)
        prediction = self.prediction(predict_input)
        self.sensors = prediction[0][-1][1::]
        self.sensors_sequence.append(self.sensors)
        self.reward = self.sensors_sequence.data[-1][0]

    def getObservation(self):
        return array(self.sensors)

    def getPoleAngles(self):
        return self.sensors[0]

    def getCartPosition(self):
        return self.sensors[2]

    def isFinished(self):
        if abs(self.getPoleAngles())> 0.7:
            # pole has fallen
            return True
        elif abs(self.getCartPosition()) > 2.4:
            # cart is out of it's border conditions
            return True
        elif self.t >= self.N:
            # maximal timesteps
            return True
        return False

    def reset(self):
        if self.randomInitialization:
            angle = random.uniform(-0.2, 0.2)
            pos = random.uniform(-0.5, 0.5)
        else:
            angle = -0.2
            pos = 0.2
        self.t = 0
        self.sensors_sequence = RingBuffer(N_CTIME_STEPS, ivalue=[0.0] * 4)
        self.actions_sequence = RingBuffer(N_CTIME_STEPS, ivalue=[0.0])
        self.sensors = (angle, 0.0, pos, 0.0)
        self.sensors_sequence.append(self.sensors)

    def getReward(self):
        return self.reward





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
N_INPUT_FEATURES = 5

# Output Features
N_ACTIONS = 1



# Length of each input sequence of data
N_TIME_STEPS = 1  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1



# SGD learning rate
LEARNING_RATE = 8e-2

# Number of iterations to train the net
N_ITERATIONS = 1000000


def one_iteration(task, all_params):
    """
    Give current value of weights, output all rewards
    :return:
    """
    rewards = []
    observations = []
    actions = []
    _all_params = lasagne.layers.get_all_params(l_action_formed)
    _all_params[0].set_value(theano_form(all_params, shape=(4, 1)))
    task.reset()
    while not task.isFinished():
        obs = task.getObservation()
        observations.append(obs)
        states = theano_form(obs, shape=[N_BATCH, 1, N_INPUT_FEATURES - 1]) # this is for each time step
        model_action_result = action_prediction(states)
        actions.append(model_action_result.reshape(1))
        task.performAction(model_action_result)
        rewards.append(task.getReward())
    last_obs = task.getObservation()
    return rewards, actions, observations, last_obs, sum(rewards)

def one_sim_iteration(task, all_params):
    """
    This function estimates the reward by
    RNN function. in our case, it is LSTM
    """

    rewards = []
    observations = []
    actions = []
    _all_params = lasagne.layers.get_all_params(l_action_formed)
    _all_params[0].set_value(theano_form(all_params, shape=(4, 1)))

    while not task.isFinished():
        obs = task.getObservation()
        observations.append(obs)
        states = theano_form(obs, shape=[N_BATCH, 1, N_INPUT_FEATURES - 1]) # this is for each time step
        model_action_result = action_prediction(states)
        actions.append(model_action_result.reshape(1))
        task.performAction(model_action_result)
        rewards.append(task.getReward())
    last_obs = task.getObservation()
    return rewards, actions, observations, last_obs, sum(rewards)


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


def chunks(lst, n):
    """ Yield successive n-sized chunks from l.
    """
    dim = len(lst[0])
    l = ([[0] * dim] * (n-1))
    l.extend(lst)
    for i in xrange(0, len(l)- n + 1, 1):
        yield l[i:i+n]


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
    input = TT.tensor3('input')
    action_prediction = T.function([input], l_action_formed.get_output(input))



    #
    N_CINPUT_FEATURES = 5

    # Critic Learning Rate
    CLEARNING_RATE = 1e-4

    # Number of time steps pf critic network
    N_CTIME_STEPS = 5

    # HIDDEN UNIT OF LSTM
    N_LSTM_HIDDEM= 4

    # Output Features
    N_OUTPUT_FEATURES = 5

    # Critic Batch
    N_CBATCH = 1

    # This is an critic model
    l_critic_in = lasagne.layers.InputLayer(shape=(N_CBATCH, N_CTIME_STEPS, N_CINPUT_FEATURES))  # Extra + 1 is from Action
    # Followed by LSTM Layer
    l_lstm_1 = lasagne.layers.LSTMLayer(input_layer=l_critic_in,
                                           num_units=N_LSTM_HIDDEM)
    l_lstm_reshape_1 = lasagne.layers.ReshapeLayer(input_layer=l_lstm_1,
                                            shape=(N_CBATCH * N_CTIME_STEPS, N_LSTM_HIDDEM))

    # Followed by a Dense Layer to Produce Output
    l_reward = lasagne.layers.DenseLayer(incoming=l_lstm_reshape_1,
                                                num_units=N_OUTPUT_FEATURES,
                                                nonlinearity=lasagne.nonlinearities.identity)
    l_reward_formed = lasagne.layers.ReshapeLayer(input_layer=l_reward,
                                        shape=(N_CBATCH, N_CTIME_STEPS, N_OUTPUT_FEATURES))


    # Cost function is mean squared error
    critic_input = TT.tensor3('critic_input')
    critic_output = TT.tensor3('critic_output')
    reward_prediction = T.function([critic_input], l_reward_formed.get_output(critic_input))
    cost = TT.mean((l_reward_formed.get_output(critic_input)[:, :, :] - critic_output[:, :, :])**2)
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(l_reward_formed), CLEARNING_RATE)
    train = T.function([critic_input, critic_output], cost, updates=updates)


    # create environment
    env = CartPoleEnvironment()
    # create task
    task = BalanceTask(env, 200, desiredValue=None)

    sim_task = SimBalanceTask(prediction=reward_prediction, maxsteps=200)



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


        previous_cost = 10000
        real_world_sample_counts = 0
        thinking_count = 0
        for n in xrange(1500):

            epsilon, epsilon_star = sample_parameter(sigmas=sigmas)
            if previous_cost < 0.08:

                rewards1, actions1, observations1, last_obs1, reward1 = one_sim_iteration(sim_task, all_params=current + epsilon)
                rewards2, actions2, observations2, last_obs2, reward2 = one_sim_iteration(sim_task, all_params=current - epsilon)
                thinking_count += 1
                if thinking_count == 5:
                    previous_cost = 1000
                    thinking_count = 0
            else:
                # Perform actions in real environment

                rewards1, actions1, observations1, last_obs1, reward1 = one_iteration(task=task, all_params=current + epsilon)
                real_world_sample_counts += 1
                if reward1 > best_reward:
                    best_reward = reward1
                rewards2, actions2, observations2, last_obs2, reward2 = one_iteration(task= task, all_params=current - epsilon)
                real_world_sample_counts += 1
                if reward2 > best_reward:
                    best_reward = reward2


                # Prepare for data for first process
                actions1 = theano_form(actions1, shape=(len(actions1), 1))
                observations1 = theano_form(observations1, shape=(len(observations1), 4))
                predicted_obs1 = concatenate([observations1[1::], [last_obs1]])
                input_data1 = concatenate([actions1, observations1], axis=1)
                output_data1 = concatenate([theano_form(rewards1, shape=(len(rewards1), 1)), predicted_obs1], axis=1)
                costs1 = []

                # Training with data gathered from first process
                critic_train_inputs = list(chunks(input_data1, N_CTIME_STEPS))
                critic_train_outputs = list(chunks(output_data1, N_CTIME_STEPS))
                for input, output in zip(critic_train_inputs, critic_train_outputs):

                    critic_train_input = theano_form(input, shape=(N_CBATCH, N_CTIME_STEPS, N_CINPUT_FEATURES))
                    critic_train_output = theano_form(output, shape=(N_CBATCH, N_CTIME_STEPS, N_OUTPUT_FEATURES))
                    costs1.append(train(critic_train_input, critic_train_output))


                # Prepare for data for second process
                actions2 = theano_form(actions2, shape=(len(actions2), 1))
                observations2 = theano_form(observations2, shape=(len(observations2), 4))
                predicted_obs2 = concatenate([observations2[1::], [last_obs2]])
                input_data2 = concatenate([actions2, observations2], axis=1)
                output_data2 = concatenate([theano_form(rewards2, shape=(len(rewards2), 1)), predicted_obs2], axis=1)
                costs2=[]

                # Training with data gathered from second process
                critic_train_inputs = list(chunks(input_data2, N_CTIME_STEPS))
                critic_train_outputs = list(chunks(output_data2, N_CTIME_STEPS))
                for input, output in zip(critic_train_inputs, critic_train_outputs):
                    critic_train_input = theano_form(input, shape=(N_CBATCH, N_CTIME_STEPS, N_CINPUT_FEATURES))
                    critic_train_output = theano_form(output, shape=(N_CBATCH, N_CTIME_STEPS, N_OUTPUT_FEATURES))
                    costs2.append(train(critic_train_input, critic_train_output))

                previous_cost = mean(costs1) + mean(costs2)


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
            current = current + LEARNING_RATE * fakt * epsilon

            if fakt2 > 0: #for sigma adaption alg. follows only positive gradients
                #apply sigma update locally
                sigmas = sigmas + LEARNING_RATE * fakt2 * (epsilon * epsilon - sigmas * sigmas) / sigmas

            arg_reward.append(mreward)




            if not n%10:
                print "previous_cost :", previous_cost
                print "real_word_example :", real_world_sample_counts
                epsilon, epsilon_star = sample_parameter(sigmas=sigmas)
                _, _, _, _, test_reward1 = one_iteration(task=task, all_params=current + epsilon)
                _, _, _, _, test_reward2 = one_iteration(task= task, all_params=current - epsilon)
                print "test_results", (test_reward1 + test_reward2)/2
                temp_arg = sum(arg_reward)/len(arg_reward)
                #records[time].append(temp_arg)
                print "best reward:", best_reward, "average reward:", temp_arg
                print
                #arg_reward = []
    #print records
    #pickle.dump(records, open("records_lambda.p", "wb"))



