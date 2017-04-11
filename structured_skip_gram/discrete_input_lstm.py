from keras.layers import Layer
import theano.tensor as T
from keras.engine import InputSpec
import theano
from keras import initializers, regularizers
from structured_skip_gram.units import param_counts, param_split
from structured_skip_gram.units import lstm_params, lstm_step
from structured_skip_gram.units import discrete_lstm_params, discrete_lstm_step


class DiscreteLSTM(Layer):
    def __init__(self, units, k,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 return_sequences=True,
                 *args, **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.units = units
        self.k = k
        self.input_spec = InputSpec(ndim=2)
        self.return_sequences = return_sequences
        Layer.__init__(self, *args, **kwargs)

    def build(self, input_shape):
        # print "Z: {}, X: {}".format(z_shape, x_shape)
        assert len(input_shape) == 2

        params_input = discrete_lstm_params(self, self.units, self.k, "input")
        params_lstm = lstm_params(self, self.units, "lstm")

        self.param_counts = param_counts((params_input, params_lstm))
        self.non_sequences = (params_input + params_lstm)
        self.built = True

    # sequence, prior, non-seq
    def step(self, x0, h0, y0, *params):
        (params_input, params_lstm) = param_split(params, self.param_counts)
        input_h = discrete_lstm_step(h0, x0, *params_input)
        h1, y1 = lstm_step(input_h, *params_lstm)
        return h1, y1

    def call(self, x):
        n = x.shape[0]
        xr = T.transpose(x, (1, 0))
        outputs_info = [T.zeros((n, self.units), dtype='float32'), T.zeros((n, self.units), dtype='float32')]
        (h, yr), _ = theano.scan(self.step, sequences=[xr], outputs_info=outputs_info,
                                 non_sequences=self.non_sequences)
        y = T.transpose(yr, (1, 0, 2))
        if self.return_sequences:
            return y
        else:
            return y[:, -1, :]

    def compute_output_shape(self, input_shapes):
        assert (len(input_shapes) == 2)
        if self.return_sequences:
            return input_shapes[0], input_shapes[1], self.units
        else:
            return input_shapes[0], self.units
