from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import cPickle as pkl

def _read_sentences(filename):
    '''Read sentence segmented PTB.'''
    with open(filename, 'r') as f:
        return map(lambda x: x.replace('\n', "<eos>"), f.readlines())

def corrupt(seq, p0, p1):
    """Delete words with p0, swap non-overlapping bi-grams with p1.

    WARNING: Not optimized for speed!

    Parameters
    ----------
    seq : list
        List of elements to delete or swap (tokens or indices)
    p0 : float [0, 1]
        Probability of dropping a word.
    p1 : float [0, 1]
        Probability of swapping non-overlapping bi-grams.
    Returns
    -------
    list
        Corrupted sequence.
    """
    # using len(seq) number of bernoulli random variables
    del_mask = np.random.binomial(1, p0, len(seq))
    seq_prime = [x for x, y in zip(seq, del_mask) if y != 1]
    if len(seq_prime) > 1:  # swapping not defined for one symbol
        swap_probs = np.random.binomial(1, p1, int(len(seq_prime)/2))
        for i in range(0, len(swap_probs)):
            if swap_probs[i] == 1:
                a = seq_prime[i*2+1]  # second element of 2-gram
                b = seq_prime[i*2]    # first element of 2-gram
                seq_prime[i*2+1] = b
                seq_prime[i*2] = a
    return seq_prime

def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    ss = [word_to_id[w] if w in word_to_id else 1  for w in data]
    return ss#[word_to_id[word] for word in data]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    dictionary = data_path + '/ptb_dict_word.pkl'
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)
    word_to_id = worddicts#_build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        #yield (x, y)
        print(x)


def ptb_sentence_iterator(data_path, p_drop, p_swap, l):
    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
      data_path: Path to the data folder.
      p_drop: float, Probability of dropping a word
      p_swap: float, Probability of swapping elements (after dropping)
      l: int, keep last l words
    """
    raw_sents = _read_sentences(os.path.join(data_path, "ptb.train.txt"))
    dictionary = data_path+'/ptb_dict_word.pkl'
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)
    word_to_id = worddicts
    for sent in raw_sents:
        s = sent.split()
        ss = np.array([word_to_id[w] if w in word_to_id else 1 for w in s])
        ss_hat = corrupt(ss[:l], p_drop, p_swap)
        yield(ss, ss_hat)

#raw = ptb_raw_data('./data')
#ptb_iterator(raw[0], 10, 50)
ptb_sentence_iterator('./data', 0.5, 0.5, 4)
