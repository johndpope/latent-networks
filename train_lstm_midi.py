#!/usr/bin/env python

import os
from lm_lstm_midi import train

def main(job_id, params):
    print(params)
    validerr = train(
        dataset=params['dataset'],
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_input=params['dim_input'][0],
        dim=params['dim'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        dim_proj=params['dim_proj'][0],
        weight_aux=params['weight_aux'][0],
        batch_size=16,
        valid_batch_size=16,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        valid_dataset=None,
        dictionary=None,
        use_dropout=params['use-dropout'][0],
        kl_start=params['kl_start'][0],
        kl_rate=0.0001)
    return validerr

if __name__ == '__main__':
    try:
        # Created experiments folder, if needed.
        os.makedirs("./experiments/midi/")
    except:
        pass

    main(0, {
        'model': ['./experiments/midi/'],
        'dataset': 'muse',
        'dim_input': [88],
        'dim': [300],
        'dim_proj': [500],
        'optimizer': ['adam'],
        'kl_start': [0.2],
        'decay-c': [0.],
        'use-dropout': [False],
        'weight_aux': [0.0],
        'learning-rate': [0.001],
        'reload': [False]})
