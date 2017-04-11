import os

os.environ["THEANO_FLAGS"] = "optimizer=None"
from structured_skip_gram.model import SSG
from structured_skip_gram.decoder_layer import DecoderLayer, DecoderLayerTest
from keras.models import Model
from keras.layers import Input
import numpy as np
def main():
    k = 10
    latent_dim = 5
    hidden_dim = 512
    z_depth = 8
    x_depth = 24
    SSG(x_k=k, y_k=k, z_depth=z_depth, latent_dim=latent_dim, hidden_dim=hidden_dim)

    decoder = DecoderLayer(256, 10)
    input_z = Input((None, latent_dim), dtype='float32')
    input_y = Input((None,), dtype='int32')
    y = decoder([input_z, input_y])
    m = Model([input_z, input_y], y)
    _z = np.random.random((64, z_depth, latent_dim)).astype(np.float32)
    _x = np.random.randint(size=(64, x_depth), low=0, high=1)
    _y = m.predict([_z, _x])
    print _y.shape
    print y.dtype
    print _y.dtype

    d2 = DecoderLayerTest(decoder)
    input_rng = Input((None,), dtype='float32')
    y2 = d2([input_z, input_rng])
    m2 = Model([input_z, input_rng], y2)
    _rng = np.random.random((64, x_depth)).astype(np.float32)
    _y2 = m2.predict([_z, _rng])
    print "Y2"
    print _y2.shape
    print y2.dtype
    print _y2.dtype
    print m2.outputs[0].dtype
if __name__ == "__main__":
    main()