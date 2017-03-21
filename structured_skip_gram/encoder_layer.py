
from keras.layers import Layer
import theano.tensor as T
from keras.engine import InputSpec
import theano
from keras import initializers, regularizers
from structured_skip_gram.units import param_counts, param_split, decoder_input_params, decoder_input_step
from structured_skip_gram.units import dense_params, dense_step, lstm_params, lstm_step, make_pair, make_W, make_b
from structured_skip_gram.units import decoder_inner_input_params, decoder_inner_input_step


class EncoderLayer(Layer):
    def __init__(self, units, k,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.units = units
        self.k = k
        self.input_spec = InputSpec(ndim=2)
        Layer.__init__(self, *args, **kwargs)

    def build(self, (z_shape, x_shape)):
        assert len(z_shape) == 2
        assert len(x_shape) == 2
        self.latent_dim = z_shape.shape[2]

        params_input = decoder_input_params(self, self.units, self.latent_dim, "outer_input")
        params_lstm = lstm_params(self, self.units, "outer_lstm")

        self.param_counts = param_counts((params_input, params_lstm, params_h, params_y,
                                          params_input_inner, params_lstm_inner, params_h_inner, params_y_inner))
        self.param_counts_inner = param_counts((params_input_inner, params_lstm_inner, params_h_inner, params_y_inner))
        self.non_sequences = (params_input + params_lstm + params_h + params_y +
                              params_input_inner + params_lstm_inner + params_h_inner + params_y_inner)
        self.built = True

    def inner_step(self, x, h0, y0, z, *params):
        (params_input_inner, params_lstm_inner,
         params_h_inner, params_y_inner) = param_split(params, self.param_counts_inner)
        hidden = decoder_inner_input_step(h0, x, z, *params_input_inner)
        h1, tmp = lstm_step(hidden, *params_lstm_inner)
        tmp = T.tanh(dense_step(tmp, params_h_inner))
        y1 = T.nnet.softmax(dense_step(tmp, params_y_inner))
        return h1, y1

    # sequence, prior, non-seq
    def step(self, z, h0, y0, x, *params):
        n = z.shape[0]
        x_depth = x.shape[0]

        (params_input, params_lstm, params_h, params_y,
         params_input_inner, params_lstm_inner, params_h_inner, params_y_inner) = param_split(params, self.param_counts)

        input_h = decoder_input_step(h0, z, *params_input)
        outer_hidden, tmp = lstm_step(input_h, *params_lstm)
        tmp = T.tanh(dense_step(tmp, *params_h))
        z_processed = T.tanh(dense_step(tmp, *params_y))
        outputs_info = [T.zeros((n, self.units), dtype='float32'),
                        T.zeros((n, self.k + 1), dtype='float32')]
        (h_inner, y_inner), _ = theano.scan(self.inner_step,
                                            sequences=[x], outputs_info=outputs_info,
                                            non_sequences=[z_processed] +
                                                          params_input_inner + params_lstm_inner +
                                                          params_h_inner + params_y_inner)

        return outer_hidden, y_inner

    def call(self, (z, x)):
        xr = T.transpose(x, (1, 0))
        zr = T.transpose(z, (1, 0, 2))
        n = z.shape[0]
        outputs_info = [T.zeros((n,))]
        (h, yr), _ = theano.scan(self.step, sequences=[zr], outputs_info=outputs_info,
                                 non_sequences=[xr] + self.non_sequences)
        # output: z depth, x depth, n, k
        y = T.transpose(yr, (2, 0, 1, 3))
        return y


class DecoderLayerTest(Layer):
    def __init__(self, decoder, *args, **kwargs):
        self.decoder = decoder
        self.units = self.decoder.units
        Layer.__init__(*args, **kwargs)

    def inner_step(self, rng, h0, x0, z, *params):
        (params_input_inner, params_lstm_inner,
         params_h_inner, params_y_inner) = param_split(params, self.decoder.inner_param_counts)
        hidden = decoder_inner_input_step(h0, x0, z, *params_input_inner)
        h1, tmp = lstm_step(hidden, *params_lstm_inner)
        tmp = T.tanh(dense_step(tmp, params_h_inner))
        y1 = T.nnet.softmax(dense_step(tmp, params_y_inner))
        cump = T.cumsum(y1, axis=1)
        choice = T.sum(rng > cump, axis=1)
        x1 = choice + 1
        x1 = T.cast(x1, "int32")
        return h1, x1

    # sequence, prior, non-seq
    def step(self, z, h0, y0, rng, *params):
        n = z.shape[0]
        x_depth = rng.shape[0]

        (params_input, params_lstm, params_h, params_y) = param_split(params, self.param_counts)

        input_h = decoder_input_step(h0, z, *params_input)
        outer_hidden, tmp = lstm_step(input_h, *params_lstm)
        tmp = T.tanh(dense_step(tmp, *params_h))
        z_processed = T.tanh(dense_step(tmp, *params_y))
        outputs_info = [T.zeros((n, self.units), dtype='float32'),
                        T.zeros((n,), dtype='int32')]
        (h_inner, y_inner), _ = theano.scan(self.inner_step,
                                            sequences=[rng],
                                            outputs_info=outputs_info,
                                            non_sequences=[z_processed])

        return outer_hidden, y_inner

    def call(self, (z, rng)):
        zr = T.transpose(z, (1, 0, 2))
        rngr = T.transpose(rng, (1, 0))
        n = z.shape[0]
        depth = rng.shape[1]
        outputs_info = [T.zeros((n, self.units), dtype='float32'), T.zeros((n, depth), dtype='int32')]
        (h, yr), _ = theano.scan(self.step,
                                 sequences=[zr],
                                 outputs_info=outputs_info,
                                 non_sequences=[rngr] + self.decoder.non_sequences)
        # output: z depth, x depth, n
        y = T.transpose(yr, (2, 0, 1))
        return y
