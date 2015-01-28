import lasagne
import theano
import einstein as E


setting = E.model.Setting()
setting.nbatches = 1
setting.learning_rate = 1e-4
setting.append_layer(lasagne.layers.InputLayer)
setting.layers[0]
setting.append_layer(lasagne.layers.LSTMLayer)
setting.append_layer(lasagne.layers.ReshapeLayer)
setting.append_layer(lasagne.layers.DenseLayer)
setting.append_layer(lasagne.layers.ReshapeLayer)


# Input features
N_INPUT_FEATURES = 4

# Output Features
N_OUTPUT_FEATURES = 1

# Length of each input sequence of data
N_TIME_STEPS = 12  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 20

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 1e-6

# Number of iterations to train the net
N_ITERATIONS = 1000000

# Forget rate
FORGET_RATE = 0.9

    
GRADIENT_METHOD = 'sgd'

model_params = [
    (lasagne.layers.InputLayer, {"shape": (N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES)}),

    (lasagne.layers.LSTMLayer, {"num_units": N_HIDDEN}),

    (lasagne.layers.ReshapeLayer, {"shape": (N_BATCH * N_TIME_STEPS, N_HIDDEN)}),

    (lasagne.layers.DenseLayer, {"num_units": N_OUTPUT_FEATURES, "nonlinearity": theano.tensor.tanh}),

    (lasagne.layers.ReshapeLayer, {"shape": (N_BATCH, N_TIME_STEPS, N_OUTPUT_FEATURES)})
    ]
    
modle = E.model.Model(model_params=model_params, n_time_steps=4)
