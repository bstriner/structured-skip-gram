import os

#os.environ["THEANO_FLAGS"] = "optimizer=None"

from structured_skip_gram.model import SSG
from structured_skip_gram.dataset import brown_words, clean_words, map_words, vectors_to_matrix, get_charset
from structured_skip_gram.dataset import batch_generator, sample_mat, decode_row
from structured_skip_gram.callbacks import write_autoencoded
import numpy as np
from keras.callbacks import LambdaCallback



def main():
    z_depth = 8
    window = 7
    batch_size = 64
    latent_dim = 2
    hidden_dim = 256
    steps_per_epoch = 128
    epochs = 1000
    autoencoded_path = "output/autoencoded-{:08d}.txt"

    words = clean_words(brown_words())
    charset, charmap = get_charset(words)
    k = len(charset)
    print "Charset: {}".format(k)
    mat = vectors_to_matrix(map_words(words, charmap))
    gen = batch_generator(mat, window, batch_size, k, z_depth)
    model = SSG(x_k=k, y_k=k, z_depth=z_depth, latent_dim=latent_dim, hidden_dim=hidden_dim)
    cb_autoencode = LambdaCallback(on_epoch_end=write_autoencoded(
        autoencoded_path,
        model.model_autoencode,
        mat,
        charset))
    model.model_train.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[cb_autoencode])


if __name__ == "__main__":
    main()
