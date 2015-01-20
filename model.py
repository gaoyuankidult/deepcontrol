__author__ = 'gao'

from layers.rec_layers import LSTMLayer
from layers.basic_layers import DenseLayer
from initializer.rand_init import RandomSparseInit
from numpy.random.mtrand import RandomState
import theano.tensor as T
from theano import function
from theano import printing
from theano import config
from theano import grad
from theano import shared
from numpy import zeros
from optimizer.grad_methods import nesterov_grad
from collections import OrderedDict

config.mode = 'FAST_COMPILE'
config.optimizer='fast_compile'
config.exception_verbosity='high'
config.on_unused_input='ignore'


class Model():
    """
    This class takes care of the modelling of system.
    """
    def __init__(self,
                 model_params):
        """
        This function is responsible for creating a model based on the dictionary
        :param o_dict: a ordered dictionary type that contains information of layers.
        First item in dictionary is the first layer. Second item in dictionary is the second layer.
        :type o_dict: collections.OrderedDict
        :return: None
        :rtype: None
        """
        self.model_params = model_params
        self.inputs = T.vector(dtype=config.floatX)
        self.targets = T.vector(dtype=config.floatX)

        model_length = len(self.model_params)
        o_iter = iter(self.model_params.items())
        k, v = o_iter.next()
        self.variable_params = []
        layer = k(**v)
        layer.inputs = self.inputs
        layer.build(**layer.build_params)
        self.variable_params.extend(layer.params)
        for _ in xrange(model_length - 1):
            previous_layer = layer
            k, v = o_iter.next()
            layer = k(**v)
            layer.inputs = previous_layer.outputs
            layer.build(**layer.build_params)
            self.variable_params.extend(layer.params)
        self.outputs = layer.outputs
        self.layer_cost = T.sum((self.outputs - self.targets)**2)
        self.build_model()

    def build_model(self):
        #self.layer_cost = T.sum((self.outputs - self.targets)**2)
        self.updates = []
        self.grads = [grad(self.layer_cost, param) for param in self.variable_params]
        nesterov_grad(self.variable_params,
                      self.grads,
                      self.updates,
                      learning_rate=1e-3,
                      momentum=0.6,
                      weight_decay=0.01)

        self.__train = function(inputs=[self.inputs, self.targets],
                                outputs=self.layer_cost,
                                updates=self.updates)

        self.__predicts = function(inputs=[self.inputs], outputs=self.outputs)

    def train(self,
              inputs, targets,
              epochs):
        for _ in xrange(epochs):
            print self.__train(inputs, targets)

    def predict(self, inputs, targets):
        pass

if __name__=="__main__":
    from numpy import arange

    rng = RandomState()
    init_cls = RandomSparseInit()


    N_MODEL_INPUTS = 1
    N_LSTM_INPUTS = 1
    N_LSTM_HIDDEN = 1
    N_LSTM_STEPS = 10
    N_OUTPUTS = 1

    model_params = OrderedDict({
        LSTMLayer: {"rng": rng,
                    "init_cls": init_cls,
                    "n_steps": N_LSTM_STEPS,
                    "n_units": N_LSTM_HIDDEN,
                    "n_in": N_MODEL_INPUTS},
        DenseLayer: {"rng": rng,
                     "init_cls": init_cls,
                     "n_in": N_LSTM_HIDDEN,
                     "n_out": N_OUTPUTS,
                     "f_act": T.tanh},
    })


    x_inputs = arange(0, 1).T
    x_outputs = arange(1, 2).T
    model = Model(model_params=model_params)
    model.train(x_inputs, x_outputs, 10)