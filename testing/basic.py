#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Custom libraries
from common.functions import read_config, debug
from preprocessings.read import extract_checkins_per_user, extract_checkins_per_venue, \
    extract_checkins_all, extract_friendships

def extract_checkins(config, dataset_name, mode, run_by):
  ### Extracting checkins
  if run_by == 'location': ### If extracted by each venue (Simplified SIGMOD 2013 version)
    checkins, grouped = extract_checkins_per_venue(dataset_name, mode, config)
  elif run_by == 'checkin': ### Map-reduce fashion but per check-in
    checkins, grouped = extract_checkins_all(dataset_name, mode, config)
  else: ### Default is by each user
    checkins, grouped = extract_checkins_per_user(dataset_name, mode, config)
  return checkins, grouped

### Testing reading the dataset and its statistics
def test1():
  ### Started the program
  debug('Started Test on SCI+')
  ### Read config
  config = read_config()
  kwargs = config['kwargs']
  run_by = kwargs['colocation']['run_by']
  all_datasets = config['dataset']
  all_modes = config['mode']
  n_core = kwargs['n_core']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    for mode in modes:
      k = all_modes.index(mode)
      debug('Run Test on Dataset', dataset_name, p, 'Mode', mode, k, '#Core', n_core)
      ### Test extract check-ins
      checkins, grouped = extract_checkins(config, dataset_name, mode, run_by)
      ### Test extract friendships
      friend_df = extract_friendships(dataset_name, config)
      debug('#friend', len(friend_df))
      uids = checkins['user'].unique()
      debug('#user', len(uids))
      locs = checkins['location'].unique()
      debug('#location', len(locs))
      pd.set_option('display.max_columns', 10)
      debug(grouped.head(30))
  debug('Finished Test on SCI+')

if __name__ == '__main__':
  test1()