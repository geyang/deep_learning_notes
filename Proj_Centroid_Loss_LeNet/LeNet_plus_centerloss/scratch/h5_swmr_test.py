#!/usr/bin/env python3

import h5py

with h5py.File('./test.h5', 'a', libver='latest', swmr=True) as f:
    print(f)