#!/usr/bin/env python

import os
from lm_lstm_blizzard import train

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
<<<<<<< HEAD
        batch_size=32,
=======
        weight_aux=params['weight_aux'][0],
        batch_size=128,
>>>>>>> 2cedae0253b87dec5dcb6e39d36bf2d811884420
        valid_batch_size=32,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
        use_dropout=params['use-dropout'][0],
<<<<<<< HEAD
        kl_start=1.0,
=======
        kl_start=params['kl_start'][0],
>>>>>>> 2cedae0253b87dec5dcb6e39d36bf2d811884420
        kl_rate=0.00005)
    return validerr

if __name__ == '__main__':
    try:
        # Created experiments folder, if needed.
<<<<<<< HEAD
        os.makedirs("./experiments/timit/")
=======
        os.makedirs("./experiments/blizzard/")
>>>>>>> 2cedae0253b87dec5dcb6e39d36bf2d811884420
    except:
        pass

    main(0, {
<<<<<<< HEAD
        'model': ['./experiments/timit/'],
        'dim_input': [200],
        'dim': [2000],
        'dim_proj': [600],
        'optimizer': ['adam'],
        'decay-c': [0.],
=======
        'model': ['./experiments/blizzard/'],
        'dim_input': [200],
        'dim': [2048],
        'dim_proj': [800],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'kl_start': [0.2],
        'weight_aux': [0.1],
>>>>>>> 2cedae0253b87dec5dcb6e39d36bf2d811884420
        'use-dropout': [False],
        'learning-rate': [0.001],
        'reload': [False]})
