from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Layer, Input, Dense, Embedding, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from structured_skip_gram.decoder_layer import DecoderLayer, DecoderLayerTest
import theano.tensor as T


# def crossentropy(ytrue, ypred):
#    return T.sum(-ytrue * T.log(ypred), axis=None)


class SSG(object):
    def __init__(self, x_k, y_k, z_depth, latent_dim=5, hidden_dim=512, lr=1e-3):
        input_x = Input((None,), dtype="int32")
        input_rng = Input((None,), dtype="float32")
        input_y = Input((None,), dtype="int32")
        input_z = Input((None, latent_dim), dtype="float32")

        embedding = Embedding(x_k + 1, hidden_dim)
        reader = LSTM(hidden_dim)
        encoder = LSTM(hidden_dim, return_sequences=True)
        encoder_z = TimeDistributed(Dense(latent_dim, activation='tanh'))
        decoder = DecoderLayer(hidden_dim, y_k)

        h = embedding(input_x)
        h = reader(h)
        h = Lambda(lambda _x: T.repeat(_x.dimshuffle((0, 'x', 1)), z_depth, 1),
                   output_shape=lambda _x: (_x[0], z_depth, _x[1]))(h)
        h = encoder(h)
        z = encoder_z(h)
        y = decoder([z, input_y])

        self.model_train = Model(inputs=[input_x, input_y], outputs=y)
        self.model_train.compile(Adam(lr), "categorical_crossentropy")

        self.model_encode = Model(inputs=[input_x], outputs=z)

        decoder_test1 = DecoderLayerTest(decoder)
        ypred = decoder_test1([input_z, input_rng])
        self.model_decode = Model(inputs=[input_z, input_rng], outputs=ypred)

        decoder_test2 = DecoderLayerTest(decoder)
        yautoencoded = decoder_test2([z, input_rng])
        self.model_autoencode = Model(inputs=[input_x, input_rng], outputs=yautoencoded)

        print "Ys: {}, {}, {}".format(y.dtype, ypred.dtype, yautoencoded.dtype)
        print "Ys: {}, {}, {}".format(y.ndim, ypred.ndim, yautoencoded.ndim)

        self.model_train.summary()
        self.model_autoencode.summary()
