#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import os
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import debug

def combine_colocation(config, p, t, d):
  debug('Started', 'p', p, 't', t, 'd', d)
  read_compressed = config['kwargs']['combine']['read_compressed']
  compress_output = config['kwargs']['combine']['compress_output']
  colocation_dir = config['directory']['colocation']
  if read_compressed is True:
    colocation_weekday_name = config['intermediate']['colocation']['compressed'].format(p, 1, t, d)
    colocation_weekend_name = config['intermediate']['colocation']['compressed'].format(p, 2, t, d)
  else:
    colocation_weekday_name = config['intermediate']['colocation']['csv'].format(p, 1, t, d)
    colocation_weekend_name = config['intermediate']['colocation']['csv'].format(p, 2, t, d)
  weekday = pd.read_csv('/'.join([colocation_dir, colocation_weekday_name]))
  weekend = pd.read_csv('/'.join([colocation_dir, colocation_weekend_name]))
  df = pd.concat([weekday, weekend])
  if compress_output is True:
    output_name = config['intermediate']['colocation']['compressed'].format(p, 0, t, d)
    compression = 'bz2'
  else:
    output_name = config['intermediate']['colocation']['csv'].format(p, 0, t, d)
    compression = None
  df.to_csv('/'.join([colocation_dir, output_name]), \
        header=True, index=False, compression=compression)
  del weekday, weekend, df
  debug('Finished', 'p', p, 't', t, 'd', d)