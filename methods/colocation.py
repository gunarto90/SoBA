#!/usr/bin/env python
"""
This code is a library: it should not run on its own.
"""
### General library
import numpy as np
import pandas as pd
import sys
import os
import math
### KD Tree
from scipy import spatial
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import IS_DEBUG, fn_timer, debug, haversine

"""
Utility functions
"""
def extract_geometry(df):
  x = df['latitude'].values
  y = df['longitude'].values
  data = np.array(zip(x.ravel(), y.ravel()))
  return data

def create_spatial_kd_tree(df):
  data = extract_geometry(df)
  tree = spatial.KDTree(data)
  return tree

def create_temporal_kd_tree(df):
  x = df['timestamp'].values
  data = zip(x.ravel())
  tree = spatial.KDTree(data)
  return tree

def normalize_radius_index(idx):
  i = 0
  out = {}
  for x in idx:
    out[i] = x
    i += 1
  return out

"""
Co-location generation
Input:
- checkins_per_user (dictionary of [int, dataframe])
- start : the beginning index of co-location generation process (useful for parallelization)
- finish : the ending index of co-location generation process (useful for parallelization)
- t_diff : time threshold for co-location criterion (in seconds)
- s_diff : spatial threshold for co-location criterion (in lat/lon degree)
"""
@fn_timer
def generate_colocation(checkins_per_user, start=0, finish=0, t_diff=1800, s_diff=0.001):
  colocations = []
  uids = checkins_per_user.keys()
  for i in range(start, finish):
    u_i = uids[i]
    df_i = checkins_per_user[u_i].sort_values(by=['timestamp'])
    debug('#Checkins of User', u_i, ':', len(df_i))
    si_tree = create_spatial_kd_tree(df_i)
    ti_tree = create_temporal_kd_tree(df_i)
    for j in range(i+1, len(uids)):
      u_j = uids[j]
      df_j = checkins_per_user[u_j].sort_values(by=['timestamp'])
      sj_tree = create_spatial_kd_tree(df_j)
      tj_tree = create_temporal_kd_tree(df_j)
      ### temporal co-occurrence
      t_idx = ti_tree.query_ball_tree(tj_tree, t_diff)
      t_count = sum(len(x) for x in t_idx)
      if t_count > 0:
        # debug(t_count,t_idx)
        ### spatial co-occurrence
        s_idx = si_tree.query_ball_tree(sj_tree, s_diff)
        s_count = sum(len(x) for x in s_idx)
        ### Only if both temporal and spatial co-occurrence > 0
        if s_count > 0:
          ### Finding the intersection and adding colocations to the list
          colocations.extend(extract_radius_search_results(u_i, u_j, df_i, df_j, s_idx, t_idx))
      ### For debugging purpose
      if IS_DEBUG is True and j > 300:
        break
  return colocations

"""
Co-location report generation (version 1)
u_i     : User i's ID
u_j     : User j's ID
df_i    : Dataframe of user i
df_j    : Dataframe of user j
s_idx   : List of list of all locations within the "radius"
t_idx   : 
"""
def extract_radius_search_results(u_i, u_j, df_i, df_j, s_idx, t_idx):
  results = []
  if len(s_idx) != len(t_idx):
    return None
  for i in range(len(s_idx)):
    temp = set(s_idx[i]).intersection(t_idx[i])
    for j in temp:
      di = df_i.iloc[i]
      dj = df_j.iloc[j]
      ### Format: user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff
      t_diff = math.fabs(di['timestamp'] - dj['timestamp'])
      s_diff = haversine(di['latitude'], di['longitude'], dj['latitude'], dj['longitude'])
      obj = (u_i, u_j, int(di['location']), int(dj['location']), di['timestamp'],dj['timestamp'],
        di['latitude'], di['longitude'], dj['latitude'], dj['longitude'],
        t_diff, s_diff
      )
      results.append(obj)
  return results