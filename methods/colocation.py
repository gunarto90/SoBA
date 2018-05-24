#!/usr/bin/env python
"""
This code is a library: it should not run on its own.
"""
### General library
import numpy as np
import pandas as pd
import sys
import os
### KD Tree
from scipy import spatial
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import IS_DEBUG, fn_timer, debug

"""
KD-Tree functions
"""
def intersect(hrect, r2, centroid):
    """
    checks if the hyperrectangle hrect intersects with the
    hypersphere defined by centroid and r2
    """
    maxval = hrect[1,:]
    minval = hrect[0,:]
    p = centroid.copy()
    idx = p < minval
    p[idx] = minval[idx]
    idx = p > maxval
    p[idx] = maxval[idx]
    return ((p-centroid)**2).sum() < r2

def radius_search(tree, datapoint, radius):
  """ find all points within radius of datapoint """
  stack = [tree[0]]
  inside = []
  while stack:
    leaf_idx, leaf_data, left_hrect, \
              right_hrect, left, right = stack.pop()
    # leaf
    if leaf_idx is not None:
      param=leaf_data.shape[0]
      distance = np.sqrt(((leaf_data - datapoint.reshape((param,1)))**2).sum(axis=0))
      near = np.where(distance<=radius)
      if len(near[0]):
        idx = leaf_idx[near]
        distance = distance[near]
        inside += (zip(distance, idx))
  else:
    if intersect(left_hrect, radius, datapoint):
      stack.append(tree[left])
    if intersect(right_hrect, radius, datapoint):
      stack.append(tree[right])
  return inside

"""
Utility functions
"""
def extract_geometry(df):
  x = df['latitude'].values
  y = df['longitude'].values
  data = zip(x.ravel(), y.ravel())
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
"""
@fn_timer
def generate_colocation(checkins_per_user, start=0, finish=0, t_diff=1800, s_diff=0.001):
  colocations = []
  uids = checkins_per_user.keys()
  for i in range(start, finish):
    u_i = uids[i]
    df_i = checkins_per_user[u_i]
    debug('User', u_i, len(df_i))
    s_tree = create_spatial_kd_tree(df_i)
    t_tree = create_temporal_kd_tree(df_i)
    for j in range(i+1, len(uids)):
      u_j = uids[j]
      df_j = checkins_per_user[u_j]
      geo_j = extract_geometry(df_j)
      time_j = zip(df_j['timestamp'].values.ravel())
      ### temporal co-occurrence
      t_idx = t_tree.query_ball_point(time_j, t_diff) ### Using temporal distance
      t_idx = normalize_radius_index(t_idx)
      # debug('t_idx', t_idx)
      # t_results, t_count = extract_radius_search_results(u_i, u_j, df_i, df_j, t_idx)
      t_count = sum(len(x) for x in t_idx.values())
      ### spatial co-occurrence
      if t_count > 0:
        s_idx = s_tree.query_ball_point(geo_j, s_diff)  ### Using spatial distance  ### Problem: Why we do not use the "haversine formula"
        s_idx = normalize_radius_index(s_idx)
        # debug('s_idx', s_idx)
        # results, count = extract_radius_search_results(u_i, u_j, df_i, df_j, s_idx)

        for k in s_idx.keys():
          temp = set(s_idx[k]).intersection(t_idx[k])
          if len(temp) > 0:
            debug(i, j, k, len(temp), temp)
      ### For debugging purpose
      # if IS_DEBUG is True and j > 10:
      #   break
  return colocations

"""
Co-location report generation (version 1)
u_i     : User i's ID
u_j     : User j's ID
df_i    : Dataframe of user i
df_j    : Dataframe of user j
idx     : List of list of all locations within the "radius"
"""
def extract_radius_search_results(u_i, u_j, df_i, df_j, idx):
  results = []
  count = 0
  geometry_i = pd.DataFrame(df_i, columns=['latitude', 'longitude', 'timestamp']).values
  geometry_j = pd.DataFrame(df_j, columns=['latitude', 'longitude', 'timestamp']).values
  for j, x in idx.items():
    count += len(x)
    if len(x) > 0:
      for i in x:
        temp = (u_i, u_j, geometry_i[i], geometry_j[j])
        results.append(temp)
  del geometry_i
  del geometry_j
  return results, count