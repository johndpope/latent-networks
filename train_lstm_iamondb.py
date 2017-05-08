#!/usr/bin/env python

import os
from lm_lstm_iamondb import train

def main(job_id, params):
    print(params)
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_input=params['dim_input'][0],
        dim=params['dim'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        dim_proj=params['dim_proj'][0],
        batch_size=16,
        valid_batch_size=32,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
        use_dropout=params['use-dropout'][0],
        kl_start=1.0,
        kl_rate=0.00005,
        weight_aux=0.0005)
    return validerr

if __name__ == '__main__':
    exp_dir = "./experiments/iamondb/"
    try:
        # Created experiments folder, if needed.
        os.makedirs(exp_dir)
    except:
        pass

    main(0, {
        'model': [exp_dir],
        'dim_input': [3],
        'dim': [1200],
        'dim_proj': [600],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})
