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

def main():
  ### Global parameter for the experiments
  ps = []     ### Active project: 0 Gowalla, 1 Brightkite
  ks = []     ### Mode for top k users: 0 Weekend, -1 All users
  ts = []     ### Time threshold
  ds = []     ### Distance threshold

  # ps.append(0)
  ps.append(1)

  ks.append(0)
  # ks.append(-1)

  ts.append(1800)
  ds.append(0)

  ### Directory and filename format
  directory = '/'.join(['./co-location', '20180517', ''])
  filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
  # co-raw: user1,user2,vid,t_diff,frequency,time1,time2,t_avg,lat,lon,distance
  out_name_checkin = 'p{}_k{}_t{}_d{}_10.checkin'
  out_name_friends = 'p{}_k{}_t{}_d{}_10.friends'
  # checkin_header = 'mid,uid,locid'
  # friends_header = 'u1,u2'

  for p in ps:
    ### Initialize variables
    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
    for k in ks:
      debug('p:{}, k:{}'.format(p, k))
      for t in ts:
        for d in ds:
          debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d), out_file=True)
          df = pd.read_csv(directory + filename.format(p, k ,t, d))
          df_friend = df[['user1', 'user2']].drop_duplicates().rename(index=str, columns={"user1": "u1", "user2": "u2"})
          df_check_in1 = df[['user1', 'vid']].rename(index=str, columns={"user1": "uid", "vid": "locid"})
          df_check_in2 = df[['user2', 'vid']].rename(index=str, columns={"user2": "uid", "vid": "locid"})
          df_check_in1.insert(0, 'mid', range(0, len(df_check_in1)))
          df_check_in2.insert(0, 'mid', range(len(df_check_in1), len(df_check_in1)+len(df_check_in2)))
          frames = [df_check_in1, df_check_in2]
          df_checkin = pd.concat(frames)
          ### Write the result to file
          df_friend.to_csv(working_folder + out_name_friends.format(p, k, t, d), index=False)
          df_checkin.to_csv(working_folder + out_name_checkin.format(p, k, t, d), index=False)

# Main function
if __name__ == '__main__':
  main()