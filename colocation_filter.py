#!/usr/bin/python

"""
This code is intended to generate a new check-in and friend dataset 
based on the users who have at least 2 co-location with other users
(Useful for testing the CCS2017 method)
###Checkin template
  mid: check-in id (auto increment number)
  uid: user id
  locid: location id
###friends template
  u1: user 1 (user id)
  u2: user 2 (user id)
"""

import pandas as pd
import numpy as np

from general_utilities import *
from base import *
from classes import *

def generate_out_name(p, k, t, d):
  name = ''
  if p == 0:
    name = 'G'
  else:
    name = 'B'
  if k == 0:
    name += 'W'
  elif k == -1:
    name += 'A'
  out_name_checkin = '{}_t{}_d{}_10.checkin'.format(name, t, d)
  out_name_friends = '{}_t{}_d{}_10.friends'.format(name, t, d)

  return out_name_checkin, out_name_friends

def main():
  ### Global parameter for the experiments
  ps = []     ### Active project: 0 Gowalla, 1 Brightkite
  ks = []     ### Mode for top k users: 0 Weekend, -1 All users
  ts = []     ### Time threshold
  ds = []     ### Distance threshold

  ps.append(0)
  ps.append(1)

  ks.append(0)
  ks.append(-1)

  ts.append(1800)
  ds.append(0)

  ### Directory and filename format
  directory = '/'.join(['./co-location', '20180518 (New on all users)', ''])
  # directory = '/'.join(['./co-location', 'test', ''])
  co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
  dataset_directory = ['gowalla', 'brightkite']
  dataset_additional_path = ['base', 'weekend']
  friendship_filename = 'friend.csv'
  dtypes = {'user1':np.int32, 'user2':np.int32, 'vid':np.int32, 't_diff':np.int32, 'frequency':np.int32, 
    'time1':np.int32, 'time2':np.int32, 't_avg':np.float64, 'lat':np.float64, 'lon':np.float64, 'distance':np.int32}
  names = ['user1','user2','vid','t_diff','frequency','time1','time2','t_avg','lat','lon','distance']
  # checkin_header = 'mid,uid,locid'
  # friends_header = 'u1,u2'

  for p in ps:
    ### Initialize variables
    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
    for k in ks:
      debug('p:{}, k:{}'.format(p, k))
      friend_ori_path = '/'.join([dataset_directory[p], dataset_additional_path[k+1], friendship_filename])
      df_friend_ori = pd.read_csv(friend_ori_path, names=['u1', 'u2'], dtype={'u1': np.int32, 'u2': np.int32})
      # debug(friend_ori_path)
      # print(df_friend_ori.describe())
      # print(df_friend_ori.head(10))
      for t in ts:
        for d in ds:
          debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d), out_file=True)
          df = pd.read_csv(directory + co_raw_filename.format(p, k ,t, d), header=0)

          ## Handles df friends
          df_friend = df[['user1', 'user2']].drop_duplicates().rename(index=str, columns={"user1": "u1", "user2": "u2"})
          # print(df_friend.describe())
          # print(df_friend.head(10))
          intersection = df_friend_ori.loc[df_friend_ori['u1'].isin(df_friend['u1'])]
          intersection = intersection.loc[intersection['u2'].isin(df_friend['u2'])]
          intersection.dropna(inplace=True)
          # print('intersection')
          # print(intersection.describe())

          ### Handles df checkins
          df_check_in1 = df[['user1', 'vid']].rename(index=str, columns={"user1": "uid", "vid": "locid"})
          df_check_in2 = df[['user2', 'vid']].rename(index=str, columns={"user2": "uid", "vid": "locid"})
          df_check_in1.insert(0, 'mid', range(1, len(df_check_in1)+1))
          df_check_in2.insert(0, 'mid', range(len(df_check_in1)+1, len(df_check_in1)+len(df_check_in2)+1))
          frames = [df_check_in1, df_check_in2]
          df_checkin = pd.concat(frames)
          ### Remove "errorenous data"
          df_checkin = df_checkin[(df_checkin['uid'] != 'user1') & (df_checkin['uid'] != 'user2') & (df_checkin['locid'] != 'vid')]
          # print(len(df_checkin))
          # print(df_friend.head())
          # print(df_checkin.head())
          ### Write the result to file
          out_name_checkin, out_name_friends = generate_out_name(p, k, t, d)
          intersection.to_csv(working_folder + out_name_friends.format(p, k, t, d), index=False)
          df_checkin.to_csv(working_folder + out_name_checkin.format(p, k, t, d), index=False)

# Main function
if __name__ == '__main__':
  main()