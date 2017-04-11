import os
from .dataset import sample_mat, decode_row
import numpy as np


def model_checkpoint(path_format, model, frequency):
    def on_epoch_end(epoch, logs):
        if (epoch + 1) % frequency == 0:
            path = path_format.format(epoch)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(path)
            model.save_weights(path)

    return on_epoch_end


def write_autoencoded(path_format, model_autoencode, mat, charset, n=32, m=16):
    def on_epoch_end(epoch, logs):
        path = path_format.format(epoch)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        words = sample_mat(mat, n)
        word_samples = [model_autoencode.predict([words, np.random.random(words.shape)]) for _ in range(m)]
        with open(path, 'w') as f:
            for i in range(n):
                word = decode_row(words[i, :], charset).strip()
                strs = ", ".join(decode_row(w[i, -1, :], charset).strip() for w in word_samples)
                f.write("{}: {}\n".format(word, strs))

    return on_epoch_end
