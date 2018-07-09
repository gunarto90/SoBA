#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
import gc
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
  config_name = 'config_test.json'
  ### Started the program
  debug('Started Test test_checkin_stats on SCI+', config_name)
  ### Read config
  config = read_config(config_name)
  kwargs = config['kwargs']
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
      checkins, _ = extract_checkins_all(dataset_name, mode, config)
      checkins['u_count'] = checkins.groupby('user')['user'].transform('count')
      df, _ = extract_checkins_all(dataset_name, mode, config, filter=False)
      df['u_count'] = df.groupby('user')['user'].transform('count')
      df = df[(df['u_count'] > 1)]
      n_user_ori = len(df['user'].unique())
      n_checkins_ori = len(df)
      n_checkins_filter = len(checkins)
      ### Test extract friendships
      friend_df = extract_friendships(dataset_name, config)
      n_friend = len(friend_df)      
      uids = checkins['user'].unique()
      n_user_filter = len(uids)      
      locs = checkins['location'].unique()
      n_locs = len(locs)
      friend_match_checkin = friend_df.isin(uids)
      friend_df = friend_df[friend_match_checkin['user1'] & friend_match_checkin['user2']]
      friend_ids = np.unique(np.concatenate((friend_df['user1'].values, friend_df['user2'].values)))
      checkins = df.loc[df['user'].isin(friend_ids)]
      n_user_friend = len(checkins['user'].unique())
      n_checkins_friend = len(checkins)
      avg_checkin_ori = n_checkins_ori/n_user_ori
      avg_checkin_filter = n_checkins_filter/n_user_filter
      avg_checkin_friend = n_checkins_friend/n_user_friend
      # debug('#user ori', n_user_ori)
      # debug('#friend', n_friend)
      # debug('#user', n_user_filter)
      # debug('#location', n_locs)
      # debug('#friend w/ checkins', n_friend)
      # debug('Avg. #Checkins ori', avg_checkin_ori)
      # debug('Avg. #Checkins filtered', avg_checkin_filter)
      # debug('#user (#friend>1)', n_user_friend)
      debug(n_user_ori, n_user_friend, n_user_filter, n_checkins_ori, n_checkins_friend, n_checkins_filter, n_friend, n_locs, avg_checkin_ori, avg_checkin_friend, avg_checkin_filter)
  debug('Finished Test on SCI+')

def test_colocation_stats():
  config_name = 'config_test.json'
  ### Started the program
  debug('Started Test test_checkin_stats on SCI+', config_name)
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
          total_user = 0
          total_friend = 0
          total_colocation = 0
          i = 0
          for colocation_df in read_colocation_file(config, p, k, t, d, chunksize=10 ** 6, usecols=['user1', 'user2']):
            colocation_df = determine_social_tie(colocation_df, friend_df)
            total_colocation += len(colocation_df)
            colocation_df = colocation_df.drop_duplicates(['user1', 'user2'])
            total_user += len(colocation_df)
            total_friend += sum(colocation_df['link'])
            i += 1
            # debug('Processing chunks #%d' % i)
          # debug('#colocations', total_colocation, '#total_user', total_user, '#total_friend', total_friend, 'p', p, 'k', k, 't', t, 'd', d)
          debug(total_colocation, total_user, total_friend, p, k, t, d)
          gc.collect()
  debug('Finished Test on SCI+')

def generating_walk2friend_data():
  config_name = 'config_test.json'
  debug('Started Test test_checkin_stats on SCI+', config_name)
  config = read_config(config_name)
  ### Started the program
  ### Read config
  config = read_config(config_name)
  kwargs = config['kwargs']
  all_datasets = config['dataset']
  all_modes = config['mode']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  directory = config['directory']['intermediate']
  for dataset_name in datasets:
    p = all_datasets.index(dataset_name)
    for mode in modes:
      k = all_modes.index(mode)
      debug('Run Test on Dataset', dataset_name, p, 'Mode', mode, k)
      ### Test extract check-ins
      checkins, _ = extract_checkins_all(dataset_name, mode, config)
      checkins.sort_values(["user", "timestamp"], inplace=True)
      checkins['mid'] = range(1, len(checkins)+1)
      checkins.rename(columns={"user": "uid", "location": "locid"}, inplace=True)
      checkins = checkins[['mid', 'uid', 'locid']]
      checkins.to_csv('/'.join([directory, '%s_%s_10.checkin' % (dataset_name, mode)]))
      ### Test extract friendships
      friend_df = extract_friendships(dataset_name, config)
      friend_df.sort_values(["user1", "user2"], inplace=True)
      friend_df.sort_values(["user1", "user2"], inplace=True)
      friend_df.rename(columns={"user1": "u1", "user2": "u2"}, inplace=True)
      friend_df.to_csv('/'.join([directory, '%s_%s_10.friends' % (dataset_name, mode)]))

if __name__ == '__main__':
  pass
  # test_checkin_stats()
  # test_colocation_stats()
  generating_walk2friend_data()