from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Layer, Input, Dense, Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from structured_skip_gram.decoder_layer import DecoderLayer, DecoderLayerTest
import theano.tensor as T
def crossentropy(ytrue, ypred):
    return T.mean(-ytrue*T.log(ypred),axis=None)

class SSG(object):
    def __init__(self, x_k, y_k, latent_dim=5, hidden_dim=512):
        input_x = Input((None,), dtype="int32")
        input_rng = Input((None,), dtype="float32")
        input_y = Input((None,), dtype="int32")
        input_z = Input((None, latent_dim), dtype="float32")


        embedding = Embedding(x_k, hidden_dim)
        encoder = LSTM(hidden_dim, return_sequences=True)
        encoder_z = TimeDistributed(Dense(latent_dim, activation='tanh'))
        decoder = DecoderLayer(hidden_dim, y_k)

        x_encoded = embedding(input_x)
        zh = encoder(x_encoded)
        z = encoder_z(zh)
        y = decoder([z, input_y])
        print "Y dim: {}".format(y.ndim)

        self.model_train = Model(inputs=[input_x, input_y], outputs=[y])
        self.model_train.compile(Adam(1e-3), crossentropy)

        self.model_encode = Model(inputs=[input_x], outputs=[z])

        decoder_test = DecoderLayerTest(decoder)
        ypred = decoder_test([input_z, input_rng])
        self.model_decode = Model(inputs=[input_z, input_rng], outputs=[ypred])

        yautoencoded = decoder_test([z, input_rng])
        self.model_autoencode = Model(inputs=[input_x, input_rng], outputs=[yautoencoded])
