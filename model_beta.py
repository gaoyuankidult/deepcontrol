import lasagne as L
import einstein as E
import einstein.data_structure as D
import theano as T
import theano.tensor as TT
import numpy as np


#Actor Network

# Network Properties
ans = E.model.Setting() # Actor Network Setting
ans.n_batches = 1
ans.learning_rate = 1e-3
ans.n_time_steps = 6
ans.n_input_features = 4
ans.n_output_features = 1
ans.n_iterations = 10000
ans.n_trans = 5
ans.serial = E.serial.socket.SocketServer()

# First Layer is Input Layer
input_layer_setting = E.model.InputLayerSetting()
input_layer_setting.shape = (ans.n_batches,
                             ans.n_time_steps,
                             ans.n_input_features)
ans.append_layer(layer=L.layers.InputLayer, layer_setting=input_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (ans.n_batches, ans.n_time_steps, ans.n_input_features)
ans.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

# Middle Layer is LSTM
lstm_layer_setting = E.model.LSTMLayerSetting()
lstm_layer_setting.num_units = 2
ans.append_layer(L.layers.LSTMLayer, lstm_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (ans.n_batches * ans.n_time_steps, lstm_layer_setting.num_units)
ans.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

# Output Layer for An Action
dense_layer_setting = E.model.DenseLayerSetting()
dense_layer_setting.num_units = 1 # only one action
dense_layer_setting.nonlinearity = L.nonlinearities.sigmoid
ans.append_layer(L.layers.DenseLayer, dense_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (ans.n_batches, ans.n_time_steps, 1)
ans.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)


# Critic Network
cns = E.model.Setting() # Critic Network Setting
cns.n_batches = 1
cns.learning_rate = 1e-3
cns.n_time_steps = 6
cns.n_input_features = 5 # network takes four features of state and one action
cns.n_output_features = 1
cns.n_iterations = 10000
cns.n_trans = 5

# First Layer is Input Layer
input_layer_setting = E.model.InputLayerSetting()
input_layer_setting.shape = (cns.n_batches,
                             cns.n_time_steps,
                             cns.n_input_features)
cns.append_layer(layer=L.layers.InputLayer, layer_setting=input_layer_setting)

lstm_layer_setting = E.model.LSTMLayerSetting()
lstm_layer_setting.num_units = 2
cns.append_layer(L.layers.LSTMLayer, lstm_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (cns.n_batches * cns.n_time_steps, lstm_layer_setting.num_units)
cns.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)

## Followed by Another Dense Layer for Reward
dense_layer_setting = E.model.DenseLayerSetting()
dense_layer_setting.num_units = cns.n_output_features
dense_layer_setting.nonlinearity = L.nonlinearities.linear
cns.append_layer(L.layers.DenseLayer, dense_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.shape = (cns.n_batches, cns.n_time_steps, cns.n_output_features)
cns.append_layer(L.layers.ReshapeLayer, reshape_layer_setting)



class ModelBetaActor(E.model.Model):
    def __init__(self, setting):
        super(ModelBetaActor, self).__init__(setting)

    def build_functions(self):
        super(ModelBetaActor,self).build_functions()

    def get_binomial_action(self, p):
        return np.random.binomial(1, p)

    def train(self, inputs, outputs):
        self.build_functions()

class ModelBetaCritic(E.model.Model):
    def __init__(self):
        super(ModelBetaCritic, self).__init__()

    def build_functions(self):
        super(ModelBetaCritic, self).build_functions()

    def train(self, inputs, outputs):
        self.build_functions()
        self.predict(input)



class ModelBeta(E.model.Model):
    def __init__(self, setting):
        self.mba = ModelBetaActor()
        self.mbc = ModelBetaCritic()
        self.setting = setting
    def train(self):
        print "sending"
        # first send n_time_steps information to the client
        self.setting.serial.send_int(self.setting.n_time_steps)
        print "sent"
        self.cost = [0] * self.setting.n_iterations
        for n in xrange(self.setting.n_iterations):
            signal = self.setting.serial.receive()
            epoch_data = signal.split(',') # rm1 is reward of last time step
            self.ring_buffer.append(epoch_data)
            buffered_data = self.ring_buffer.get()
            if None not in buffered_data:
                all_data = D.theano_form(list=buffered_data, shape=[self.setting.n_batches,
                                                                    self.setting.n_time_steps+1,
                                                                    self.setting.n_trans])

                actor_train_inputs = all_data[:, 0:self.setting.n_time_steps, 1::]
                action_predict = self.mba.predict(actor_train_inputs)
                critic_train_inputs = action_predict[:, :, :] + actor_train_inputs
                critic_train_outputs = all_data[
                                :,
                                1::, # extract reward from 1 to N_TIME_STEPS,
                                0].reshape([self.setting.n_batches,
                                            self.setting.n_time_steps,
                                            self.setting.n_output_features])# Reward takes the first position
                self.mbc.train(inputs=critic_train_inputs, outputs=critic_train_outputs)
















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
                                                                    self.setting._n_trans])

                train_inputs = all_data[:, 0:self.setting.n_time_steps, 1::]
                model_reward_result = self.pred_reward(train_inputs)
                # set desired output, the second number of result is reward
                train_outputs = all_data[
                                :,
                                1::, # extract reward from 1 to N_TIME_STEPS,
                                0].reshape([self.setting.n_batches,
                                            self.setting.n_time_steps,
                                            self.setting.n_output_features])# Reward takes the first position
                self.costs[n] = self._train(train_inputs, train_outputs)
                if not n % 10:
                    cost_val = self.compute_cost(train_inputs, train_outputs)
                    print "Iteration {} validation cost = {}".format(n, cost_val)
                    print "reward predict: "
                    print model_reward_result
                    print "train results:"
                    print train_outputs
                # Extract the most recent action from all result.
                action = self.get_binomial_action(self.pred_action(train_inputs)[:, -1]) * 2 - 1
                #print "prob", self.pred_action(train_inputs)[:, -1]
                self.setting.serial.send_int(action)

model = ModelBetaActor(ans)
model.train()