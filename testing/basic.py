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
    extract_checkins_all, extract_friendships, read_colocation_file, determine_social_tie

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
def test_checkin_stats():
  ### Started the program
  debug('Started Test test_checkin_stats on SCI+')
  ### Read config
  config = read_config('config_test.json')
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
      debug(grouped.describe())
      friend_match_checkin = friend_df.isin(uids)
      friend_df = friend_df[friend_match_checkin['user1'] & friend_match_checkin['user2']]
      debug('#friend w/ checkins', len(friend_df))
  debug('Finished Test on SCI+')

def test_colocation_stats():
  ### Started the program
  debug('Started Test test_checkin_stats on SCI+')
  config = read_config('config_test.json')
  ### Read config
  config = read_config('config_test.json')
  kwargs = config['kwargs']
  all_datasets = config['dataset']
  all_modes = config['mode']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  t_diffs = kwargs['ts']
  s_diffs = kwargs['ds']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    friend_df = extract_friendships(dataset_name, config)
    for mode in modes:
      k = all_modes.index(mode)
      for t in t_diffs:
        for d in s_diffs:
          colocation_df = read_colocation_file(config, p, k, t, d)
          colocation_df = determine_social_tie(colocation_df, friend_df)
          colocation_df = colocation_df[['user1', 'user2', 'link']].drop_duplicates(['user1', 'user2'])
          debug('#colocations', len(colocation_df), sum(colocation_df['link']), 'p', p, 'k', k, 't', t, 'd', d)
  debug('Finished Test on SCI+')

if __name__ == '__main__':
  pass
  # test_checkin_stats()
  test_colocation_stats()