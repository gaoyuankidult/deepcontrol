import theano as T
from collections import Sequence
from itertools import chain, count
from numpy import array


def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=T.config.floatX).reshape(shape)


def check_list_depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def make_chunks(lst, n):
    """ Yield successive n-sized chunks from l.
    """
    dim = len(lst[0])
    l = ([[0] * dim] * (n-1))
    l.extend(lst)
    for i in xrange(0, len(l)- n + 1, 1):
        yield l[i:i+n]