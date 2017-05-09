import os
import numpy as np
import cPickle
import itertools


def load_text8(data_dir, level='char', create_pkl=True):
    '''
    Load the text8 dataset.
    '''
    if create_pkl:
        # get paths to raw text files
        tr_file = os.path.join(data_dir, 'text8.train.txt')
        te_file = os.path.join(data_dir, 'text8.test.txt')

        tr_text = open(tr_file, 'r').read()
        te_text = open(te_file, 'r').read()

        word2id = {'__pad__': 0, '__go__': 1}
        id2word = ['__pad__', '__go__']
        word2tf = {}

        tr_toks, te_toks = ([], [])

        # scan text files and convert to lists of int-valued keys
        for f_text, f_toks in [(tr_text, tr_toks),
                               (te_text, te_toks)]:
            if (level == 'word'):
                f_text = f_text.split()
            for tok in f_text:
                if not (tok in word2tf):
                    word2tf[tok] = 0
                word2tf[tok] += 1

        # sort word ids by frequency
        mc_wrd = sorted(word2tf.items(), key=lambda x: x[1], reverse=True)
        for wrd, _ in mc_wrd:
            word2id[wrd] = len(id2word)
            id2word.append(wrd)

        for f_text, f_toks in [(tr_text, tr_toks),
                               (te_text, te_toks)]:
            if (level == 'word'):
                f_text = f_text.split()
            for tok in f_text:
                assert (tok in word2id), 'something went wrong'
                f_toks.append(word2id[tok])

        # build valid dataset ~ 1% of training tokens
        tr_len = int(len(tr_toks) * 0.99)
        va_toks = tr_toks[tr_len:]
        tr_toks = tr_toks[:tr_len]

        print('-- created t8 dataset, vocabulary: {}, tr_toks: {}, va_toks: {}, te_toks: {}'.format(
              len(word2id), len(tr_toks), len(va_toks), len(te_toks)))

        # dict for easy handling
        data_dict = {'train': np.asarray(tr_toks),
                     'valid': np.asarray(va_toks),
                     'test': np.asarray(te_toks),
                     'word2idx': word2id,
                     'idx2word': id2word}

        # dump the processed data for easier reloading
        with open(os.path.join(data_dir, 't8_dataset_{}.pkl'.format(level)), 'wb') as f:
            cPickle.dump(data_dict, f, protocol=-1)

    else:
        # load preprocessed data
        with open(os.path.join(data_dir, 't8_dataset_{}.pkl'.format(level)), 'rb') as f:
            data_dict = cPickle.load(f)

    return data_dict


class Text8():
    def __init__(self, data_path, seq_len, batch_size, level="word", rng_seed=1234):
        # load ptb word sequence
        text8_data = load_text8(data_path, level=level, create_pkl=True)

        self.tr_words = text8_data['train']
        self.va_words = text8_data['valid']
        self.te_words = text8_data['test']
        self.word2idx = text8_data['word2idx']
        self.idx2word = text8_data['idx2word']

        print('idx2word:')
        print(", ".join(["{}".format(v) for v in self.idx2word[:50]]))
        print('tr_words.shape: {}'.format(self.tr_words.shape))
        print('va_words.shape: {}'.format(self.va_words.shape))
        print('te_words.shape: {}'.format(self.te_words.shape))

        self.batch_size = batch_size
        self.voc_size = len(self.idx2word)  # # of possible words
        self.seq_len=seq_len           # length of input sequences
        self.pad_id = self.word2idx['__pad__']
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)

    def _sample_subseqs(self, source_seq, seq_count, seq_len):
        '''
        Sample subsequences of the given source sequence.
        '''
        source_len = source_seq.shape[0]
        max_start_idx = source_len - seq_len
        # sample the "base" sequences
        start_idx = self.rng.randint(low=0, high=max_start_idx, size=(seq_count,))
        idx_seqs = []
        for i in range(seq_count):
            subseq = source_seq[start_idx[i]:(start_idx[i] + seq_len)]
            idx_seqs.append(subseq[np.newaxis, :])
        idx_seqs = np.vstack(idx_seqs)
        return idx_seqs.astype("int64")

    def _prepare_heldout_set(self, source_seq):
        '''build validation/test set
        '''
        seg_size = (source_seq.shape[0] + self.batch_size - 1) / self.batch_size
        source_seq = source_seq.reshape((1, source_seq.shape[0]))
        # split into nbatch lines
        x = [source_seq[:, i:i + seg_size] for i in range(0, source_seq.shape[1], seg_size)]
        # handle padding of the last segment
        x_last = np.zeros((1, seg_size), dtype='int64') + self.pad_id
        x_last[:, :x[-1].shape[1]] = x[-1]
        x[-1] = x_last
        # concatenate in nbatch lines
        x = np.concatenate(x, axis=0)
        # start with 0 vector (first token prediction)
        x = np.concatenate([np.zeros((self.batch_size, 1)), x], axis=1)
        x = x.astype("int64")
        Xva = x[:, :-1]
        Yva = x[:, 1:]
        # split into batches
        X = [Xva[:, i:i + self.seq_len] for i in range(0, Xva.shape[1], self.seq_len)]
        Y = [Yva[:, i:i + self.seq_len] for i in range(0, Yva.shape[1], self.seq_len)]
        return X, Y

    def _prepare_training_set(self, source_seq, n_batches=500):
        '''
        Prepare a sample from the source sequence.
        '''
        # sample a batch of one-hot subsequences from the source sequence
        x = self._sample_subseqs(source_seq, self.batch_size, self.seq_len * n_batches)
        x = np.concatenate([np.zeros((self.batch_size, 1)), x], axis=1)
        x = x.astype("int64")
        x_in = x[:, :-1]
        y_in = x[:, 1:]
        X = [x_in[:, i:i + self.seq_len] for i in range(0, x_in.shape[1], self.seq_len)]
        Y = [y_in[:, i:i + self.seq_len] for i in range(0, y_in.shape[1], self.seq_len)]
        assert (len(X) == n_batches)
        assert (len(Y) == n_batches)
        return X, Y

    def get_train_batch(self):
        return iter(itertools.izip(self._prepare_training_set(self.tr_words)))

    def get_valid_batch(self):
        return iter(itertools.izip(self._prepare_heldout_set(self.va_words)))

    def get_test_batch(self):
        return iter(itertools.izip(self._prepare_heldout_set(self.te_words)))
