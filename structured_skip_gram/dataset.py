from nltk.corpus import brown, shakespeare
import itertools
import numpy as np


def brown_words():
    return brown.words()


def shakespeare_words():
    """
    Concatenate all of shakespeare
    :return:
    """
    return itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids())


def clean_word(word):
    """
    Remove non-asci characters and downcase
    :param word:
    :return:
    """
    return "".join([c for c in word.lower() if ord(c) < 128])


def clean_words(words):
    """
    Remove words < 3 characters
    :param words:
    :return:
    """
    return [clean_word(w) for w in words if len(clean_word(w)) > 0]


def get_charset(words):
    """
    List unique characters
    :param words:
    :return: list of characters, dictionary from characters to indexes
    """
    charset = list(set(itertools.chain.from_iterable(words)))
    charset.sort()
    charmap = {c: i for i, c in enumerate(charset)}
    return charset, charmap


def map_word(word, charmap):
    """
    Convert string to list of indexes into charset
    :param word:
    :param charmap:
    :return:
    """
    return [charmap[c] for c in word]


def map_words(words, charmap):
    return [map_word(w, charmap) for w in words]


def decode_vector(vector, charset):
    """
    List of indexes to a string
    :param vector:
    :param charset:
    :return:
    """
    return "".join(charset[x] for x in vector)


def vector_to_array(vector, depth):
    ar = [c + 1 for c in vector]
    while len(ar) < depth:
        ar.append(0)
    return np.array(ar).reshape((1, -1))


def vectors_to_matrix(vectors):
    depth = max(len(v) for v in vectors)
    return np.vstack(vector_to_array(v, depth) for v in vectors)


def decode_row(row, charset):
    """
    Output vector to a string
    :param row:
    :param charset:
    :return:
    """
    return "".join([charset[x - 1] if x > 0 else " " for x in row])


def decode_output(output, charset):
    """
    Output matrix to list of strings
    :param output:
    :param charset:
    :return:
    """
    return [decode_row(row, charset) for row in output]


def sample_word(vectors):
    i = np.random.randint(0, len(vectors))
    return vectors[i]


def sample_words(vectors, n):
    return [sample_word(vectors) for _ in range(n)]


def sample_mat(mat, n):
    idx = np.random.randint(0, mat.shape[0], (n,))
    return mat[idx, :]


def sample_bigram(mat, window):
    if window == 0:
        ind = np.random.randint(0, mat.shape[0])
        word = mat[ind:ind + 1, :]
        return word, word
    else:
        value = np.random.randint(1, window + 1)
        sign = np.random.randint(0, 2) * 2 - 1
        offset = value * sign
        indmin = max(0, -offset)
        indmax = min(mat.shape[0] - 1, mat.shape[0] - 1 - offset)
        ind1 = np.random.randint(indmin, indmax + 1)
        ind2 = ind1 + offset
        return mat[ind1:ind1 + 1, :], mat[ind2:ind2 + 1, :]


def sample_bigrams(mat, window, n):
    grams = [sample_bigram(mat, window) for _ in range(n)]
    x = np.vstack(g[0] for g in grams)
    y = np.vstack(g[1] for g in grams)
    return x, y


def batch_generator(mat, window, n, k, z_depth):
    while True:
        x, y = sample_bigrams(mat, window, n)
        ytarg = one_hot_2d(y, k)
        ytarg = np.repeat(np.expand_dims(ytarg, 1), z_depth, 1)
        yield [x, y], ytarg


def one_hot_2d(x, x_k):
    """
    Efficient one-hot encoding of 2d vector
    :param x:
    :param x_k:
    :return:
    """
    ret = np.zeros((x.shape[0], x.shape[1], x_k + 1), dtype=np.float32)
    r = np.repeat(np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=np.int32).reshape((-1, 1)), x.shape[1], axis=1)
    c = np.repeat(np.linspace(0, x.shape[1] - 1, x.shape[1], dtype=np.int32).reshape((1, -1)), x.shape[0], axis=0)
    ret[r.ravel(), c.ravel(), x.ravel()] = 1
    return ret
