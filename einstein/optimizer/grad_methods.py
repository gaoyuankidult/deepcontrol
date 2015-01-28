__author__ = 'gao'

from theano import config
from theano import shared
from numpy import zeros

def nesterov_grad(params,
                  grads,
                  updates,
                  learning_rate = 1e-3,
                  momentum=0.6,
                  weight_decay=0.01):

    for param_i, grad_i in zip(params, grads):
        mparam_i = shared(zeros(param_i.get_value().shape, dtype=config.floatX))
        full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
