'''
Build a simple neural language model using GRU units
'''
from __future__ import print_function

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
from ipdb import set_trace as dbg
import numpy
import sys

import matplotlib.pyplot as plt


profile = False
seed = 1234
numpy.random.seed(seed)


def main():
    model_file = sys.argv[1]
    opts = model_file[:-len("_model.npz")] + "_opts.pkl"
    model_options = pkl.load(open(opts, 'rb'))

    # Load data
    data_path = './datasets/iamondb/'
    from iamondb import IAMOnDB
    iamondb = IAMOnDB(name='train',
                      prep='normalize',
                      cond=False,
                      path=data_path)
    X_mean = iamondb.X_mean
    X_std = iamondb.X_std


    print('Loading model')
    from lm_lstm_iamondb import init_params, init_tparams, load_params
    params = init_params(model_options)
    params = load_params(model_file, params)
    tparams = init_tparams(params)

    trng = RandomStreams(42)
    from lm_lstm_iamondb import build_sampler, gen_sample
    f_next = build_sampler(tparams, model_options, trng)
    sample, sample_score = gen_sample(tparams, f_next, model_options, maxlen=200, argmax=False)

    from iamondb_utils import plot_lines_iamondb_example
    plot_lines_iamondb_example(sample[0], offsets_provided=True,
                               mean=X_mean, std=X_std, colored=True)
    print("NLL: {}".format(sample_score))
    plot_lines_iamondb_example(sample[0], show=True)

    dbg()


if __name__ == '__main__':
    main()
