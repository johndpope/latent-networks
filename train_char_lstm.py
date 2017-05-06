#!/usr/bin/env python

from lm_char_lstm import train


def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=100,
        batch_size=32,
        valid_batch_size=32,
        validFreq=2000,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset='/data/lisatmp4/anirudhg/ptb/ptb_train_50w.txt',
        valid_dataset='/data/lisatmp4/anirudhg/ptb/ptb_valid.txt',
        dictionary='/data/lisatmp4/anirudhg/latent_autoregressive/char2.npz',
        use_dropout=params['use-dropout'][0],
        kl_start = 0.2,
        kl_rate =  0.00005)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/data/lisatmp4/anirudhg/latent_autoregressive/char_models/'],
        'dim_word': [620],
        'dim': [1000],
        'n-words': [50],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})
