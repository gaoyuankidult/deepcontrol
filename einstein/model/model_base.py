import lasagne as L
import theano as T
import theano.tensor as TT
import einstein as e
import einstein.data_structure as d
from collections import namedtuple


class Model():
    """
    This class takes care of the modelling of system.
    """
    def __init__(self, setting):
        """
        This function is responsible for creating a model based on the dictionary
        :param model_params: a ordered dictionary type that contains information of layers.
        First item in dictionary is the first layer. Second item in dictionary is the second layer.
        :type model_params: collections.OrderedDict
        :return: None
        :rtype: None
        """
        # Get input as internal representation
        self.setting = setting
        self.model_params = setting.get_layers_params()

        # Initialize symbolic parameters
        self.__init_symb()

        # Initialize parameters needed for later calculation
        self.current_layer = None
        self.previous_layer = None

        # Initialize communication socket
        self.ring_buffer = d.RingBuffer(size=self.setting.n_time_steps + 1)  # need reward of next step for training

        # Build the model
        self.__build_model()
        self.__build_cost_function()
        self.__build_training_rule()
        self.__build_functions()

    def __init_symb(self):
        """
        Initialize the symbolic variables of the model (e.g. input and output)
        :return:
        """
        self.input = TT.tensor3('input')
        self.target_output = TT.tensor3('target_output')

    def __build_model(self):
        """
        Use parameters of the layers to build model.
        :return:
        """
        # Parameters for storing all layers
        self.layers = []
        # Instantiate all layer
        for layer, params in self.model_params:

            # Test whether this is the first layer or not
            if None == self.previous_layer:
                self.current_layer = layer(**params)
            else:
                self.current_layer = layer(input_layer=self.previous_layer, **params)

            # Append current layer into layer list
            self.layers.append(self.current_layer)
            self.previous_layer = self.current_layer

    def __build_cost_function(self):
        if self.setting._cost_f == None:
            self.cost = TT.mean((self.layers[-1].get_output(self.input)[:, :, :]
                    - self.target_output[:, :, :])**2)
        else:
            self.cost = self.cost_f(self.layers[-1].get_output(self.input)[:, :, :], self.target_output[:, :, :])

    def __build_training_rule(self):
        # Use NAG for training
        all_params = L.layers.get_all_params(self.layers[-1])
        self.updates = L.updates.nesterov_momentum(self.cost, all_params, self.setting.learning_rate)

    def __build_functions(self):
        self._train = T.function([self.input, self.target_output], self.cost, updates=self.updates)
        self.y_pred_reward = T.function([self.input], self.layers[-1].get_output(self.input))
        self.compute_cost = T.function([self.input, self.target_output], self.cost)

    def train(self):
        # first send n_time_steps information to the client
        self.setting.serial.send_int(setting.n_time_steps)
        for n in xrange(self.setting.n_iterations):
            signal = self.setting.serial.receive()
            epoch_data = signal.split(',') # rm1 is reward of last time step
            self.ring_buffer.append(epoch_data)
            buffered_data = self.ring_buffer.get()
            if None not in buffered_data:
                all_data = d.theano_form(list=buffered_data, shape=[self.setting.n_batches,
                                                                    self.setting.n_time_steps+1,
                                                                    self.setting._n_trans])

                train_inputs = all_data[:, 0:self.setting.n_time_steps, 1::]
                model_reward_result = self.y_pred_reward(train_inputs)
                # set desired output, the second number of result is reward
                train_outputs = all_data[
                                :,
                                1::, # extract reward from 1 to N_TIME_STEPS,
                                0].reshape([self.setting.n_batches,
                                            self.setting.n_time_steps,
                                            self.setting._n_output_features])# Reward takes the first position
                self.costs[n] = self.train(train_inputs, train_outputs)
                if not n % 10:
                    cost_val = self.compute_cost(train_inputs, train_outputs)
                    print "Iteration {} validation cost = {}".format(n, cost_val)
                    print "reward predict: ", model_reward_result
                    print "train results:", train_outputs

                # Extract the most recent action from all result.
                action = 1
                self.serial.send_int("%d\0"%action)


class ModelSetting(object):
    def __init__(self, n_batches=None, learning_rate=None, time_steps=None, n_input_features=None,
                 n_output_features =None, cost_f=None, n_iterations=None, n_trains=None, serial=None):
        self._n_batches = n_batches
        self._learning_rate = learning_rate
        self._n_time_steps = time_steps
        self._n_input_features = n_input_features
        self._n_output_features = n_output_features
        self._cost_f = cost_f
        self._n_iterations = n_iterations
        # Number of transmitted variables
        self._n_trans = n_trains
        self._serial=serial


    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert isinstance(value, float)
        self._learning_rate = value

    @property
    def n_batches(self):
        return self._n_batches

    @n_batches.setter
    def n_batches(self, value):
        assert isinstance(value, int)
        self._n_batches = value

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @n_time_steps.setter
    def n_time_steps(self, value):
        assert isinstance(value, int)
        self._n_time_steps = value

    @property
    def n_input_features(self):
        return self._n_input_features

    @n_input_features.setter
    def n_input_features(self, value):
        assert isinstance(value, int)
        self._n_input_features = value

    @property
    def cost_f(self):
        return self._cost_f

    @cost_f.setter
    def cost_f(self, value):
        assert hasattr(value, '__call__')
        self._cost_f = value

    @property
    def n_iterations(self):
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, value):
        assert isinstance(value, int)
        self._n_iterations = value

    @property
    def serial(self):
        return self.serial

    @serial.setter
    def serial(self, value):
        assert (value == e.seiral.socket.SocketServer)
        self._serial = value


class LayerSetting(object):

    @staticmethod
    def iter_properties_of_class(cls):
        for varname in vars(cls):
            value = getattr(cls, varname)
            if isinstance(value, property):
                yield varname

    def properties(self):
        result = {}
        for cls in self.__class__.mro():
            for varname in self.iter_properties_of_class(cls):
                result[varname] = getattr(self, varname)
        return result

class InputLayerSetting(LayerSetting):
    def __init__(self, shape=None):
        super(InputLayerSetting, self).__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple), "The input is " + value + "However, we restrict input type to be int."
        self._shape = value


class LSTMLayerSetting(LayerSetting):
    def __init__(self, n_lstm_hidden_units=None):
        super(LSTMLayerSetting, self).__init__()
        self._n_lstm_hidden_units = n_lstm_hidden_units

    @property
    def num_units(self):
        return self._n_lstm_hidden_units

    @num_units.setter
    def num_units(self, value):
        assert isinstance(value, int)
        self._n_lstm_hidden_units = value


class ReshapeLayerSetting(LayerSetting):
    def __init__(self, reshape_shape=None):
        super(ReshapeLayerSetting, self).__init__()
        self._shape = reshape_shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        self._shape = value

class DenseLayerSetting(LayerSetting):
    def __init__(self, dense_n_hidden_units=None):
        super(DenseLayerSetting, self).__init__()
        self._dense_n_hidden_units = dense_n_hidden_units

    @property
    def num_units(self):
        return self._dense_n_hidden_units

    @num_units .setter
    def num_units(self, value):
        assert isinstance(value, int)
        self._dense_n_hidden_units = value


class Setting(ModelSetting):
    def __init__(self):
        super(Setting, self).__init__()
        self.layers = []
        self.Layer = namedtuple('Layer', ['object', 'setting'], verbose=False)

    def l2s_map(self, layer):
        return {L.layers.InputLayer: InputLayerSetting,
                L.layers.ReshapeLayer: ReshapeLayerSetting,
                L.layers.LSTMLayer: LSTMLayerSetting,
                L.layers.DenseLayer: DenseLayerSetting}[layer]

    def append_layer(self, layer, layer_setting):
        for k, v in layer_setting.properties().items():
            if v is None:
                raise ValueError," The value of %s ->" % k + "is _None in layer" + layer.__name__
        self.layers.append(self.Layer(object=layer, setting=layer_setting))

    def get_layers_params(self):
        return [(layer, layer_setting.properties()) for layer, layer_setting in self.layers]





















