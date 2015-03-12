from lasagne.layers import shape
import numpy as np
import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import linspace
from numpy import sin as np_sin
from numpy.random import choice
from numpy.random import binomial
from serial.socket import SocketServer

np.set_printoptions()

# Number of transmitted variables
N_TRANS = 5

# Input features
N_INPUT_FEATURES = 4

# Output Features
N_ACTIONS = 1

# Output Features
N_OUTPUT_FEATURES = 5

# Length of each input sequence of data
N_TIME_STEPS = 3  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 8

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 1e-2

# Number of iterations to train the net
N_ITERATIONS = 5000000

# Forget rate
FORGET_RATE = 0.9

# Number of reward output
N_REWARD = 1

# Sampling Times
N_ACTION_SAMPLES = 1   # If we only have one simulator, we can only sample once.



def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=theano.config.floatX).reshape(shape)

if __name__ == "__main__":
    import signal
    from einstein.data_structure import RingBuffer
    import lasagne as L

    # Actor Network
    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer
    l_actor_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES))
    # Followed by LSTM Layer
    l_lstm_1 = lasagne.layers.LSTMLayer(input_layer=l_actor_in,
                                          num_units=N_HIDDEN)
    l_lstm_reshape_1 = lasagne.layers.ReshapeLayer(input_layer=l_lstm_1,
                                            shape=(N_BATCH * N_TIME_STEPS, N_HIDDEN))
        # Followed by a Dense Layer to Produce Action
    l_hidden_1 = lasagne.layers.DenseLayer(incoming=l_lstm_reshape_1,
                                                num_units=N_HIDDEN,
                                                nonlinearity=L.nonlinearities.tanh)

    # Followed by a Dense Layer to Produce Action
    l_action = lasagne.layers.DenseLayer(incoming=l_hidden_1,
                                                num_units=N_ACTIONS,
                                                nonlinearity=L.nonlinearities.sigmoid)
    l_actor_output = lasagne.layers.ReshapeLayer(input_layer=l_action,
                                        shape=(N_BATCH, N_TIME_STEPS, N_ACTIONS))



    # Critic Network
    l_critic_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES + 1))  # Extra + 1 is from Action
    # Followed by LSTM Layer
    l_lstm_2 = lasagne.layers.LSTMLayer(input_layer=l_critic_in,
                                           num_units=N_HIDDEN)
    l_lstm_reshape_2 = lasagne.layers.ReshapeLayer(input_layer=l_lstm_2,
                                            shape=(N_BATCH * N_TIME_STEPS, N_HIDDEN))
    # Followed by a Dense Layer to Produce Output
    l_reward = lasagne.layers.DenseLayer(incoming=l_lstm_reshape_2,
                                                num_units=N_OUTPUT_FEATURES,
                                                nonlinearity=L.nonlinearities.identity)
    l_reward_formed = lasagne.layers.ReshapeLayer(input_layer=l_reward,
                                        shape=(N_BATCH, N_TIME_STEPS, N_OUTPUT_FEATURES))





    # Setup Symbolic Variables of Actor Network
    actor_input = T.tensor3('actor_input', dtype=theano.config.floatX)
    actor_output = T.tensor3('actor_output', dtype=theano.config.floatX)
    # Cost = mean squared error, starting from delay point
    actor_cost = T.mean((l_actor_output.get_output(actor_input)[:, :, :]
                         - actor_output[:, :, :])**2)
    # Setup Training Process for Actor Network
    # Use NAG for training
    actor_all_params = lasagne.layers.get_all_params(l_actor_output)
    actor_updates = lasagne.updates.nesterov_momentum(actor_cost, actor_all_params, LEARNING_RATE)

    # Theano Functions for Actor Network,
    actor_train = theano.function([actor_input, actor_output], actor_cost, updates=actor_updates)
    # Predict Action
    actor_prediction = theano.function([actor_input], l_actor_output.get_output(actor_input))
    # Compute the cost
    actor_cost = theano.function([actor_input, actor_output], actor_cost)

    # Record all costs of the Actor Network.
    actor_costs = np.zeros(N_ITERATIONS)




    # Setup Symbolic Variable of Critic Network
    critic_input = T.tensor3('critic_input', dtype=theano.config.floatX)
    critic_output = T.tensor3('critic_output', dtype=theano.config.floatX)
    # Cost = mean squared error, starting from delay point
    critic_cost = T.mean((l_reward_formed.get_output(critic_input)[:, :, :]
                   - critic_output[:, :, :])**2)
    # Use NAG for training
    critic_all_params = lasagne.layers.get_all_params(l_reward_formed)
    critic_updates = lasagne.updates.nesterov_momentum(critic_cost, critic_all_params, LEARNING_RATE)

    # Theano Functions for Critic Network,
    critic_train = theano.function([critic_input, critic_output], critic_cost, updates=critic_updates)
    # Predict Action
    critic_prediction = theano.function([critic_input], l_reward_formed.get_output(critic_input))
    # Compute the cost
    critic_cost = theano.function([critic_input, critic_output], critic_cost)
    # Record all costs of the Actor Network.
    critic_costs = np.zeros(N_ITERATIONS)





    # Initialize serial communication class
    serial = SocketServer()
    ring_buffer = RingBuffer(size=N_TIME_STEPS + 1) # need reward of next step for training
    actions_set = RingBuffer(size=N_TIME_STEPS)
    actions_set.data = binomial(1, 0.5, N_TIME_STEPS).astype(theano.config.floatX).tolist()
    iter_init_actions = iter(actions_set.data)
    costs = [0] * N_ITERATIONS


    # Send n_time_steps information to client
    serial.send("%d\0" % N_TIME_STEPS)

    # Form forget vector
    forget_vector = array([FORGET_RATE**i for i in xrange(N_TIME_STEPS)])

    for n in range(N_ITERATIONS):
        if None in ring_buffer.get():
            signal = serial.receive()
            epoch_data = signal.split(',') # rm1 is reward of last time step
            ring_buffer.append(epoch_data)
            buffered_data = ring_buffer.get()
            # We can not start training if there is no enough data
        if None not in buffered_data:

            # For Evaluation of Actor Network
            all_data = theano_form(list=buffered_data, shape=[N_BATCH, N_TIME_STEPS + 1, N_TRANS])
            train_actor_inputs = all_data[:,
                                 1::,  # Use all N_TIME_STEPS for estimation,  It does not need first time step to predict something
                                 1::]   # State variables are from third position.
            # Input states and get action
            actor_result = actor_prediction(train_actor_inputs)
            # Sample Value from a
            p = actor_result[:, -1, 0]  # Extract the most recent action from all result.
            action = binomial(N_ACTION_SAMPLES, p, 1)[0]
            # Send Action to Simulator
            serial.send("%d\0" % action)


            #Update Action Set
            actions_set.append(action)
            # Receive and Analyze Reward
            # print "Receiving"
            signal = serial.receive()
            # print "Received"
            epoch_data = signal.split(',')  # rm1 is reward of last time step
            ring_buffer.append(epoch_data)
            buffered_data = ring_buffer.get()











            # For Evaluation Critic Network
            # Form the Input of Critic Network
            all_data = theano_form(list=buffered_data, shape=[N_BATCH, N_TIME_STEPS + 1, N_TRANS])
            train_critic_state_inputs = all_data[
                                  :,
                                  0:N_TIME_STEPS,  # Now We Received New Data,  Data contains action
                                  1::]  # The Data We Need from this Data Set is State
            actions = theano_form(actions_set.get(), shape=(N_BATCH, N_TIME_STEPS, N_ACTION_SAMPLES))
            train_critic_inputs = np.concatenate([actions, train_critic_state_inputs], axis=2)  # The Data We Need from this Data Set is State



            # The output number needs indexes to specify the which number is needed and which is not.
            # Form Critic Output
            train_critic_outputs = all_data[
                                   :,
                                   1::,  # extract reward from 1 to N_TIME_STEPS + 1,
                                   :].reshape([N_BATCH, N_TIME_STEPS, N_OUTPUT_FEATURES])  # Reward takes the first position

            # Train the Critic Network
            costs[n] = critic_train(train_critic_inputs, train_critic_outputs)
            reward_est = critic_prediction(train_critic_inputs)[
                         :,  # For All N_BATCHES
                         :,  # Prediction Starts from First Time Step
                         0]  # Reward is stored at first position
            train_actor_outputs = theano_form([action
                                         if reward > 0 else 1 - action
                                         for reward, action in
                                         zip(reward_est.reshape(N_TIME_STEPS),
                                             actions.reshape(N_TIME_STEPS))], shape=(N_BATCH,
                                                                                     N_TIME_STEPS,
                                                                                     1))  #Only One Output for Actor Network
            # Training of Actor Network
            actor_train(train_actor_inputs, train_actor_outputs)



            if not n % 200:
                print "Estimated reward:", reward_est
                print "Real reward and state:", train_critic_outputs[:, :, 0]
                print "Actor Network Result", actor_result
                print "Actor Samples", actions_set.get()
                print "Action Training indicator", train_actor_outputs
                #raw_input()
        else:
            serial.send("%d\0" % iter_init_actions.next())








    import matplotlib.pyplot as plt
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    plt.savefig('img.png')
