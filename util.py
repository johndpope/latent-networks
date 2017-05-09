import numpy as np
import os
import theano
import theano.tensor as T
import time
import six
from theano.ifelse import ifelse
from collections import OrderedDict
import warnings

logit = lambda u: T.log(u / (1.-u))
logit_np = lambda u: np.log(u / (1.-u)).astype(theano.config.floatX)

import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy.lib.stride_tricks import as_strided

srng = RandomStreams(seed=4884)
def totuple(arg):
    if type(arg) is not tuple:
        if isinstance(arg, list):
            return tuple(arg)
        else:
            return (arg,)
    return arg
class Iterator(object):
    """
    Dataset iterator
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, data, batch_size=None, nbatch=None,
                 start=0, end=None, shuffle=False, infinite_data=0,
                 pseudo_n=1000000):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.infinite_data = infinite_data
        if not infinite_data:
            self.start = start
            self.end = data.num_examples() if end is None else end
            if self.start >= self.end or self.start < 0:
                raise ValueError("Got wrong value for start %d." % self.start)
            self.nexp = self.end - self.start
            if nbatch is not None:
                self.batch_size = int(np.float(self.nexp / float(nbatch)))
                self.nbatch = nbatch
            elif batch_size is not None:
                self.batch_size = batch_size
                self.nbatch = int(np.float(self.nexp / float(batch_size)))
            self.shuffle = shuffle
        else:
            self.pseudo_n = pseudo_n
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        if self.infinite_data:
            for i in xrange(self.pseudo_n):
                yield self.data.slices()
        else:
            if self.shuffle:
                self.data.shuffle()
            start = self.start
            end = self.end - self.end % self.batch_size
            for idx in xrange(start, end, self.batch_size):
                yield [self.data.slices(idx, idx + self.batch_size), self.data.slices(idx + 1, idx + self.batch_size +1)]


def complex_to_real(X):
    """
    WRITEME
    Parameters
    ----------
    X : list of complex vectors
    Notes
    -----
    This function assumes X as 2D
    """
    new_X = []
    for i in xrange(len(X)):
        x = X[i]
        new_x = np.concatenate([np.real(x), np.imag(x)])
        new_X.append(new_x)
    return np.array(new_X)


def floatX(num):
    if theano.config.floatX == 'float32':
        return numpy.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))


def downscale_images(X, LEVEL):
    X = floatX(X)/floatX(LEVEL)
    return X

def upscale_images(X, LEVEL):
    X = numpy.uint8(X*LEVEL)
    return X

def stochastic_binarize(X):
    return (numpy.random.uniform(size=X.shape) < X).astype('float32')

def sample_from_softmax(softmax_var):
    #softmax_var assumed to be of shape (batch_size, num_classes)
    old_shape = softmax_var.shape

    softmax_var_reshaped = softmax_var.reshape((-1,softmax_var.shape[softmax_var.ndim-1]))

    return T.argmax(
        T.cast(
            srng.multinomial(pvals=softmax_var_reshaped),
            theano.config.floatX
            ).reshape(old_shape),
        axis = softmax_var.ndim-1
        )


def get_norms(model, gradients):
    """Compute norm of weights and their gradients divided by the number of elements"""
    norms = []
    grad_norms = []
    for param_name, param in model.params.iteritems():
        norm = T.sqrt(T.sum(T.square(param))) / T.prod(param.shape.astype(theano.config.floatX))
        norm.name = 'norm_' + param_name
        norms.append(norm)
        grad = gradients[param]
        grad_norm = T.sqrt(T.sum(T.square(grad))) / T.prod(grad.shape.astype(theano.config.floatX))
        grad_norm.name = 'grad_norm_' + param_name
        grad_norms.append(grad_norm)
    return norms, grad_norms

def create_log_dir(args, model_id):
    model_id += args.suffix + time.strftime('-%y%m%dT%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.output_dir), model_id)
    os.makedirs(model_dir)
    return model_dir


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
        return [vv for kk, vv in six.iteritems(tparams)]

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.001, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

def _p(pp, name):
    return '%s_%s' % (pp, name)


def norm_clip(dW, max_l2_norm=10.0):
    """
    Clip theano symbolic var dW to have some max l2 norm.
    """
    dW_l2_norm = T.sqrt(T.sum(dW**2.0))
    norm_ratio = (max_l2_norm / dW_l2_norm)
    clip_factor = ifelse(T.lt(norm_ratio, 1.0), norm_ratio, 1.0)
    dW_clipped = dW * clip_factor
    return dW_clipped


def gradient_clipping(grads, tparams, clip_c=1.0):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()

    g2 = T.sqrt(g2)
    not_finite = T.or_(T.isnan(g2), T.isinf(g2))
    new_grads = []

    for p, g in zip(tparams.values(), grads):
        new_grads.append(T.switch(g2 > clip_c,
                                       g * (clip_c / g2),
                                       g))

    return new_grads, not_finite, T.lt(clip_c, g2)



def get_param_updates(params=None, grads=None, \
        alpha=None, beta1=None, beta2=None, it_count=None, \
        mom2_init=1e-3, smoothing=1e-6, max_grad_norm=10000.0):
    """
    This update has some extra inputs that aren't used. This is just so it
    can be called interchangeably with "ADAM" updates.
    """

    # make an OrderedDict to hold the updates
    updates = OrderedDict()
    # alpha is a shared array containing the desired learning rate
    lr_t = alpha[0]
    for p in params:
        # get gradient for parameter p
        grad_p = norm_clip(grads[p], max_grad_norm)

        # initialize first-order momentum accumulator
        mom1_ary = 0.0 * p.get_value(borrow=False)
        mom1 = theano.shared(mom1_ary)

        # update momentum accumulator
        mom1_new = (beta1[0] * mom1) + ((1. - beta1[0]) * grad_p)

        # do update
        p_new = p - (lr_t * mom1_new)

        # apply updates to
        updates[p] = p_new
        updates[mom1] = mom1_new
    return updates

#load params
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in six.iteritems(zipped):
        new_params[kk] = vv.get_value()
    return new_params

from fuel.datasets import SVHN
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

def create_streams(train_set, valid_set, test_set, training_batch_size,
                   monitoring_batch_size):
    """Creates data streams for training and monitoring.
    Parameters
    ----------
    train_set : :class:`fuel.datasets.Dataset`
        Training set.
    valid_set : :class:`fuel.datasets.Dataset`
        Validation set.
    test_set : :class:`fuel.datasets.Dataset`
        Test set.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.
    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.
    """
    main_loop_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, training_batch_size))
    train_monitor_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, monitoring_batch_size))
    valid_monitor_stream = DataStream.default_stream(
        dataset=valid_set,
        iteration_scheme=ShuffledScheme(
            valid_set.num_examples, monitoring_batch_size))
    test_monitor_stream = DataStream.default_stream(
        dataset=test_set,
        iteration_scheme=ShuffledScheme(
            test_set.num_examples, monitoring_batch_size))

    return (main_loop_stream, train_monitor_stream, valid_monitor_stream,
            test_monitor_stream)


def create_svhn_streams(training_batch_size, monitoring_batch_size):
    """Creates SVHN data streams.
    Parameters
    ----------
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.
    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.
    """
    train_set = SVHN(2, ('train',), sources=('features',),
                     subset=slice(0, 63257))
    valid_set = SVHN(2, ('train',), sources=('features',),
                     subset=slice(63257, 73257))
    test_set = SVHN(2, ('test',), sources=('features',))

    return create_streams(train_set, valid_set, test_set, training_batch_size,
                          monitoring_batch_size)

def create_gaussian_mixture_data_streams(batch_size, monitoring_batch_size,
                                         means=None, variances=None, priors=None,
                                         rng=None, num_examples=100000,
                                         sources=('features', )):
    train_set = GaussianMixture(num_examples=num_examples, means=means,
                                variances=variances, priors=priors,
                                rng=rng, sources=sources)

    valid_set = GaussianMixture(num_examples=num_examples,
                                means=means, variances=variances,
                                priors=priors, rng=rng, sources=sources)

    main_loop_stream = DataStream(
        train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, batch_size=batch_size, rng=rng))

    train_monitor_stream = DataStream(
        train_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    valid_monitor_stream = DataStream(
        valid_set, iteration_scheme=ShuffledScheme(5000, batch_size, rng=rng))

    return main_loop_stream, train_monitor_stream, valid_monitor_stream

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald and has been discussed
    on the ML.
    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue : object
        The value to use for end='pad'
    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    use as_strided
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be "\
                          "positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' "\
                          "mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)

def save_params(params, filename, symlink=None):
    """Save the parameters.
    Saves the parameters as an ``.npz`` file. It optionally also creates a
    symlink to this archive.
    """
    np.savez(filename, **params)
    if symlink:
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(filename, symlink)
