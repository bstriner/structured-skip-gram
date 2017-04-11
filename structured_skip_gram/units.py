import theano.tensor as T
import numpy as np


def param_counts(params):
    return [len(p) for p in params]


def param_split(params, counts):
    counts = np.array(counts)
    idx_end = np.cumsum(counts)
    assert len(params) == idx_end[-1], "Expected {} params, got {}".format(idx_end[-1], len(params))
    idx_start = np.concatenate(([0], idx_end[:-1]))
    ret = [list(params[s:e]) for s, e in zip(idx_start, idx_end)]
    return ret


def make_W(layer, shape, name):
    return layer.add_weight(shape,
                            initializer=layer.kernel_initializer,
                            name=name,
                            regularizer=layer.kernel_regularizer)


def make_b(layer, shape, name):
    return layer.add_weight(shape,
                            initializer=layer.bias_initializer,
                            name=name,
                            regularizer=layer.bias_regularizer)


def make_pair(layer, shape, name):
    return [make_W(layer, shape, "{}_W".format(name)), make_b(layer, (shape[1],), "{}_b".format(name))]


def lstm_params(layer, dim, name):
    f_W, f_b = make_pair(layer, (dim, dim), "{}_f".format(name))
    i_W, i_b = make_pair(layer, (dim, dim), "{}_i".format(name))
    c_W, c_b = make_pair(layer, (dim, dim), "{}_c".format(name))
    o_W, o_b = make_pair(layer, (dim, dim), "{}_o".format(name))
    return [f_W, f_b, i_W, i_b, c_W, c_b, o_W, o_b]


def lstm_step(h_0, f_W, f_b, i_W, i_b, c_W, c_b, o_W, o_b):
    f = T.nnet.sigmoid(T.dot(h_0, f_W) + f_b)
    i = T.nnet.sigmoid(T.dot(h_0, i_W) + i_b)
    c = T.tanh(T.dot(h_0, c_W) + c_b)
    o = T.nnet.sigmoid(T.dot(h_0, o_W) + o_b)
    h_1 = h_0 * f + c * i
    y_1 = o * h_1
    return h_1, y_1


def dense_params(layer, shape, name):
    W, b = make_pair(layer, shape, name)
    return [W, b]


def dense_step(h_0, W, b):
    return T.dot(h_0, W) + b


def decoder_input_params(layer, units, latent_dim, name):
    params_input = [make_W(layer, (units, units), "{}_W".format(name)),  # hidden
                    make_W(layer, (latent_dim, units), "{}_U".format(name)),  # hidden representation
                    make_b(layer, (units,), '{}_b'.format(name))]
    return params_input


def decoder_input_step(h_0, z, W, U, b):
    return T.tanh(T.dot(h_0, W) + T.dot(z, U) + b)


def decoder_inner_input_params(layer, units, k, name):
    params_input = [make_W(layer, (units, units), "{}_W".format(name)),  # hidden
                    make_W(layer, (k + 2, units), "{}_U".format(name)),  # previous character
                    make_W(layer, (units, units), "{}_V".format(name)),  # hidden representation
                    make_b(layer, (units,), '{}_b'.format(name))]
    return params_input


def decoder_inner_input_step(h_0, x_0, z, W, U, V, b):
    return T.tanh(T.dot(h_0, W) + U[x_0, :] + T.dot(z, V) + b)

def discrete_lstm_params(layer, units, k, name):
    params_input = [make_W(layer, (units, units), "{}_W".format(name)),  # hidden
                    make_W(layer, (k, units), "{}_U".format(name)),  # discrete input
                    make_b(layer, (units,), '{}_b'.format(name))]
    return params_input


def discrete_lstm_step(h_0, x_0, W, U, b):
    return T.tanh(T.dot(h_0, W) + U[x_0, :] + b)

