import lasagne as L
import einstein as E
import einstein.data_structure as D
import theano as T
import theano.tensor as TT
import numpy as np

setting = E.model.Setting()
setting.n_batches = 1
setting.learning_rate = 1e-4
setting.n_time_steps = 6
setting.n_input_features = 4
setting.n_output_features = 1
setting.n_iterations = 100000
setting.n_trans = 5
setting.serial = E.serial.socket.SocketServer()


# First Layer is Input Layer
input_layer_setting = E.model.InputLayerSetting()
input_layer_setting.shape = (setting.n_batches,
                             setting.n_time_steps,
                             setting.n_input_features)
setting.append_layer(layer=L.layers.InputLayer, layer_setting=input_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (setting.n_batches, setting.n_time_steps, setting.n_input_features)
setting.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

# Middle Layer is LSTM
lstm_layer_setting = E.model.LSTMLayerSetting()
lstm_layer_setting.num_units = 100
setting.append_layer(L.layers.LSTMLayer, lstm_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (setting.n_batches * setting.n_time_steps, lstm_layer_setting.num_units)
setting.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

## Middle layer is Action Layer
dense_layer_setting = E.model.DenseLayerSetting()
dense_layer_setting.num_units = 1  # only one action
dense_layer_setting.nonlinearity = L.nonlinearities.sigmoid
setting.append_layer(L.layers.DenseLayer, dense_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (setting.n_batches, setting.n_time_steps, 1)
setting.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

# Followed by Another LSTM Layer
lstm_layer_setting = E.model.LSTMLayerSetting()
lstm_layer_setting.num_units = 100
setting.append_layer(L.layers.LSTMLayer, lstm_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (setting.n_batches * setting.n_time_steps, lstm_layer_setting.num_units)
setting.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)


## Followed by Another Dense Layer for Reward
dense_layer_setting = E.model.DenseLayerSetting()
dense_layer_setting.num_units = setting.n_output_features
dense_layer_setting.nonlinearity = L.nonlinearities.identity
setting.append_layer(L.layers.DenseLayer, dense_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (setting.n_batches, setting.n_time_steps, setting.n_output_features)
setting.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)



class ModelAlpha(E.model.Model):
    def __init__(self, setting):
        super(ModelAlpha, self).__init__(setting)

    def build_functions(self):
        super(ModelAlpha, self).build_functions()
        self.pred_action = T.function([self.input], self.layers[-5].get_output(self.input))

    def get_binomial_action(self, p):
        return np.random.binomial(1, p)

    def train(self):
        self.build_functions()
        print "sending"
        # first send n_time_steps information to the client
        self.setting.serial.send_int(self.setting.n_time_steps)
        print "sent"
        self.costs = [0] * self.setting.n_iterations
        for n in xrange(self.setting.n_iterations):
            signal = self.setting.serial.receive()
            epoch_data = signal.split(',') # rm1 is reward of last time step
            self.ring_buffer.append(epoch_data)
            buffered_data = self.ring_buffer.get()
            if None not in buffered_data:
                all_data = D.theano_form(list=buffered_data, shape=[self.setting.n_batches,
                                                                    self.setting.n_time_steps+1,
                                                                    self.setting.n_trans])

                train_inputs = all_data[:, 0:self.setting.n_time_steps, 1::]


                # Set desired output, the second number of result is reward
                train_outputs = all_data[
                                :,
                                1::,  # extract reward from 1 to N_TIME_STEPS,
                                0   # reward is the first element in this structure
                                ].reshape([self.setting.n_batches,
                                            self.setting.n_time_steps,
                                            self.setting.n_output_features])# Reward takes the first position
                self.costs[n] = self._train(train_inputs, train_outputs)
                # Extract the most recent action from all result.
                action = self.get_binomial_action(self.pred_action(train_inputs)[:, -1]) * 2 - 1

                self.setting.serial.send_int(action)
                if not n % 10:
                    cost_val = self.compute_cost(train_inputs, train_outputs)
                    model_reward_result = self.predict(train_inputs)
                    print "Iteration {} validation cost = {}".format(n, cost_val)
                    print "reward predict: "
                    print model_reward_result
                    print "train results:"
                    print train_outputs
                    print "predcted action"
                    print self.pred_action(train_inputs)[:, -1]


model = ModelAlpha(setting)
model.train()