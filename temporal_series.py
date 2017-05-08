#import ipdb
import numpy as np

from multiprocessing import Process, Queue
class Data(object):
    """
    Abstract class for data
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name=None, path=None, multi_process=0):
        self.name = name
        self.data = self.load(path)
        self.multi_process = multi_process
        if multi_process > 0:
            self.queue = Queue(2**15)
            processes = [None] * multi_process
            for mid in xrange(multi_process):
                processes[mid] = Process(target=self.multi_process_slices,
                                         args=(mid,))
                processes[mid].start()

    def multi_process_slices(self, mid=-1):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.multi_process_slices.")

    def load(self, path):
        return np.load(path)

    def slices(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.slices.")

    def num_examples(self):
        return max(mat.shape[0] for mat in self.data)

    def theano_vars(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.theano_vars.")


class TemporalSeries(Data):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.
    Parameters
    ----------
    .. todo::
    """
    def slices(self, start, end):
        return (mat[start:end].swapaxes(0, 1)
                for mat in self.data)

    def create_mask(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        for i, sample_len in enumerate(samples_len):
            mask[:sample_len, i] = 1.
        return mask

    def zero_pad(self, batch):
        max_sample_len = max(len(sample) for sample in batch)
        rval = np.zeros((len(batch), max_sample_len, batch[0].shape[-1]),
                        dtype=batch[0].dtype)
        for i, sample in enumerate(batch):
            rval[i, :len(sample)] = sample
        return rval.swapaxes(0, 1)

    def create_mask_and_zero_pad(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        if batch[0].ndim == 1:
            rval = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        else:
            rval = np.zeros((max_sample_len, len(batch), batch[0].shape[1]),
                            dtype=batch[0].dtype)
        for i, (sample, sample_len) in enumerate(zip(batch, samples_len)):
            mask[:sample_len, i] = 1.
            if batch[0].ndim == 1:
                rval[:sample_len, i] = sample
            else:
                rval[:sample_len, i, :] = sample
        return rval, mask

