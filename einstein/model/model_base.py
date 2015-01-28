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
    def __init__(self, model_params, n_time_steps, cost_f=None,):
        """
        This function is responsible for creating a model based on the dictionary
        :param model_params: a ordered dictionary type that contains information of layers.
        First item in dictionary is the first layer. Second item in dictionary is the second layer.
        :type model_params: collections.OrderedDict
        :return: None
        :rtype: None
        """
        # Get input as internal representation
        self.model_params = model_params
        self.n_time_steps = n_time_steps
        self.cost_f = cost_f

        # Initialize symbolic parameters
        self.__init_symb()

        # Initialize parameters needed for later calculation
        self.current_layer = None
        self.previous_layer = None

        # Initialize communication socket
        self.socket = e.serial.socket.SocketServer()
        self.ring_buffer = d.RingBuffer(size=n_time_steps + 1) # need reward of next step for training

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
        if self.cost_f == None:
            self.cost = TT.mean((self.layers[-1].get_output(self.input)[:, :, :]
                    - self.target_output[:, :, :])**2)
        else:
            self.cost = self.cost_f(self.layers[-1].get_output(self.input)[:, :, :], self.target_output[:, :, :])

    def __build_training_rule(self):
        # Use NAG for training
        all_params = L.layers.get_all_params(self.layers[-1])
        self.updates = L.updates.nesterov_momentum(self.cost, all_params, LEARNING_RATE)

    def __build_functions(self):
        self._train = T.function([self.input, self.target_output], self.cost, updates=self.updates)
        self.y_pred_reward = T.function([input], self.layers[-1].get_output(input))
        self.compute_cost = T.function([input, self.target_output], self.cost)

    def train(self):
        ring_buffer = d.RingBuffer(size=self.n_ + 1) # need reward of next step for training

class ModelSetting(object):
    def __init__(self, n_batches=None, learning_rate=None):
        self.__n_batches = n_batches
        self.__learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        return self.learning_rate = value

    @property
    def n_batches(self):
        return self.__n_batches

    @n_batches.setter
    def n_batches(self, value):
        assert isinstance(value, int)
        self.__n_batches = value



class InputLayerSetting(object):
    def method_name(self, n_input_features=None):
        self.__n_input_features = n_input_features

    def __init__(self,
                 n_input_features,
                 ):
        self.__n_input_features = n_input_features

    @property
    def n_input_feature(self):
        return self.__n_input_features

    @n_input_feature.setter
    def n_input_features(self, value):
        assert isinstance(value, int)
        self.__n_input_features = value


class LSTMLayerSetting(object):
    def __init__(self, n_lstm_hidden_units=None):
        self.__n_lstm_hidden_units = n_lstm_hidden_units

    @property
    def n_hidden_units(self):
        return self.__n_lstm_hidden_units

    @n_hidden_units.setter
    def n_hidden_units(self, value):
        assert isinstance(value, int)
        self.__n_lstm_hidden_units = value


class ReshapeLayerSetting(object):
    def __init__(self, reshape_shape=None):
        self.__reshape_shape = reshape_shape

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, int)
        self.__reshape_shape = value

class DenseLayerSetting(object):
    def __init__(self, dense_n_hidden_units=None):
        self.__dense_n_hidden_units = dense_n_hidden_units

    @property
    def dense_n_hidden_units(self):
        return self.__dense_n_hidden_units

    @dense_n_hidden_units.setter
    def dense_n_hidden_units(self, value):
        assert isinstance(value, int)
        self.__dense_n_hidden_units= value


class Setting(ModelSetting):
    def __init__(self):
        self.layers = []
        self.Layer = namedtuple('Layer', ['object', 'setting'], verbose=True)

    def l2s_map(self, layer):
        return {L.layers.InputLayer: InputLayerSetting,
                L.layers.ReshapeLayer: ReshapeLayerSetting,
                L.layers.LSTMLayer: LSTMLayerSetting,
                L.layers.DenseLayer: DenseLayerSetting}[layer]

    def append_layer(self, layer):
        print self.l2s_map(layer)
        #self.current_setting = self.l2s_map(layer)
        #for v in self.current_setting.locals():
        #    if v is None:
        #        raise ValueError," The value " + v + " in layer " + layer + " is not given."
        #self.layers.append(self.Layer(object=layer, setting=self.current_setting))








































