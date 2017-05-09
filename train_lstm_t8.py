#!/usr/bin/env python

import os
from lm_lstm_t8 import train

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
        weight_aux=params['weight_aux'][0],
        batch_size=32,
        valid_batch_size=32,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
        use_dropout=params['use-dropout'][0],
        kl_start=params['kl_start'][0],
        kl_rate=0.00005)
    return validerr

if __name__ == '__main__':
    try:
        # Created experiments folder, if needed.
        os.makedirs("./experiments/t8/")
    except:
        pass

    main(0, {
        'model': ['./experiments/t8/'],
        'dim_input': [44372],
        'dim': [500],
        'dim_proj': [300],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'kl_start': [0.2],
        'weight_aux': [0.1],
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})
