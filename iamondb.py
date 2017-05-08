import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano.tensor as T

from temporal_series import TemporalSeries
from util import segment_axis, tolist, totuple

from iamondb_utils import fetch_iamondb


class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data
    """
    def norm_normalize(self, X, avr_norm=None):
        """
        Unify the norm of each sequence in X
        Parameters
        ----------
        X       : list of lists or ndArrays
        avr_nom : Scalar
        """
        if avr_norm is None:
            avr_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                avr_norm += euclidean_norm
            avr_norm /= len(X)
        else:
            X = [x[i] / avr_norm for x in X]
        return X, avr_norm

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance
        Parameters
        ----------
        X      : list of lists or ndArrays
        X_mean : Scalar
        X_std  : Scalar
        Notes
        -----
        Compute varaince using the relation
        >>> Var(X) = E[X^2] - E[X]^2
        """
        if X_mean is None or X_std is None:
            X_len = np.array([len(x) for x in X]).sum()
            X_mean = np.array([x.sum() for x in X]).sum() / X_len
            X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
            X_std = np.sqrt(X_sqr - X_mean**2)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        """
        Standardize X such that X \in [0, 1]
        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if X_max is None or X_min is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)

    def numpy_rfft(self, X):
        """
        Apply real FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.rfft(x) for x in X])
        return X

    def numpy_irfft(self, X):
        """
        Apply real inverse FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.irfft(x) for x in X])
        return X

    def rfft(self, X):
        """
        Apply real FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.rfft(x) for x in X])
        return X

    def irfft(self, X):
        """
        Apply real inverse FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.irfft(x) for x in X])
        return X

    def stft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fft(x) for x in X])
        return X

    def istft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.real(scipy.ifft(x)) for x in X])
        return X

    def fill_zero1D(self, x, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list or 1D ndArray
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if mode == 'lefthand':
            new_x = np.concatenate([np.zeros((pad_len)), x])
        elif mode == 'righthand':
            new_x = np.concatenate([x, np.zeros((pad_len))])
        elif mode == 'random':
            new_x = np.concatenate(
                [np.zeros((pad_len)), x, np.zeros((pad_len))]
            )
        return new_x

    def fill_zero(self, X, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list of ndArrays or lists
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if pad_len == 0:
            X_max = np.array([len(x) for x in X]).max()
            new_X = np.zeros((len(X), X_max))
            for i, x in enumerate(X):
                free_ = X_max - len(x)
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((free_)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((free_))], axis=1)
                elif mode == 'random':
                    j = np.random.randint(free_)
                    new_x = np.concatenate(
                        [np.zeros((j)), x, np.zeros((free_ - j))],
                        axis=1
                    )
                new_X[i] = new_x
        else:
            new_X = []
            for x in X:
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((pad_len)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((pad_len))], axis=1)
                elif mode == 'random':
                    new_x = np.concatenate(
                        [np.zeros((pad_len)), x, np.zeros((pad_len))],
                         axis=1
                    )
                new_X.append(new_x)
        return new_X

    def reverse(self, X):
        """
        Reverse each sequence of X
        Parameters
        ----------
        X       : list of ndArrays or lists
        """
        new_X = []
        for x in X:
            new_X.append(x[::-1])
        return new_X


class IAMOnDB(TemporalSeries, SequentialPrepMixin):
    """
    IAMOnDB dataset batch provider
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', cond=False, X_mean=None, X_std=None,
                 bias=None, **kwargs):

        self.prep = prep
        self.cond = cond
        self.X_mean = X_mean
        self.X_std = X_std
        self.bias = bias

        super(IAMOnDB, self).__init__(**kwargs)

    def load(self, data_path):

        if self.name == "train":
            X, y = fetch_iamondb(data_path, subset="train")
            print("train")
            print(len(X))
            print(len(y))
        elif self.name == "valid":
            X, y = fetch_iamondb(data_path, subset="valid")
            print("valid")
            print(len(X))
            print(len(y))

        raw_X = X
        raw_X0 = []
        offset = True
        raw_new_X = []

        for item in raw_X:
            if offset:
                raw_X0.append(item[1:, 0])
                raw_new_X.append(item[1:, 1:] - item[:-1, 1:])
            else:
                raw_X0.append(item[:, 0])
                raw_new_X.append(item[:, 1:])

        raw_new_X, self.X_mean, self.X_std = self.global_normalize(raw_new_X, self.X_mean, self.X_std)
        new_x = []

        for n in range(raw_new_X.shape[0]):
            new_x.append(np.concatenate((raw_X0[n][:, None], raw_new_X[n]),
                                        axis=-1).astype('float32'))
        new_x = np.array(new_x)

        if self.prep == 'none':
            X = np.array(raw_X)

        if self.prep == 'normalize':
            X = new_x
            print(X[0].shape)
        elif self.prep == 'standardize':
            X, self.X_max, self.X_min = self.standardize(raw_X)

        self.labels = [np.array(y)]

        return [X]

    def theano_vars(self):

        if self.cond:
            return [T.ftensor3('x'), T.fmatrix('mask'),
                    T.ftensor3('y'), T.fmatrix('label_mask')]
        else:
            return [T.ftensor3('x'), T.fmatrix('mask')]

    def theano_test_vars(self):
        return [T.ftensor3('y'), T.fmatrix('label_mask')]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        label_batches = [mat[start:end] for mat in self.labels]
        mask = self.create_mask(batches[0])
        batches = [self.zero_pad(batch) for batch in batches]
        label_mask = self.create_mask(label_batches[0])
        label_batches = [self.zero_pad(batch) for batch in label_batches]

        if self.cond:
            return totuple([batches[0], mask, label_batches[0], label_mask])
        else:
            return totuple([batches[0], mask])

    def generate_index(self, X):

        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)

        return idx

if __name__ == "__main__":

    data_path = './datasets/iamondb/'
    iamondb = IAMOnDB(name='valid',
                      prep='normalize',
                      cond=False,
                      path=data_path)

    batch = iamondb.slices(start=0, end=10)
    from ipdb import set_trace as dbg
    dbg()
    X = iamondb.data[0]
    sub_X = X

    for item in X:
        max_x = np.max(item[:,1])
        max_y = np.max(item[:,2])
        min_x = np.min(item[:,1])
        min_y = np.min(item[:,2])

    print np.max(max_x)
    print np.max(max_y)
    print np.min(min_x)
    print np.min(min_y)
    ipdb.set_trace()
