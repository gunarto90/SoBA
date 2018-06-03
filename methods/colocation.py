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
import csv
import io
import re
import time
import gc
### KD Tree
from scipy import spatial
sys.setrecursionlimit(10000)  ### To make sure recursion limit do not stop the KD-tree
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import IS_DEBUG, fn_timer, debug, haversine, make_sure_path_exists, remove_file_if_exists, report_progress

colocation_header = 'user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff\n'

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
- checkins: dictionary of [int, dataframe]
- config  : configuration file
- p       : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k       : mode for dataset (0: all, 1: weekday, 2: weekend)
- t_diff  : time threshold for co-location criterion (in seconds)
- s_diff  : spatial threshold for co-location criterion (in lat/lon degree) (1 degree = 111 Km)
- start   : the beginning index of co-location generation process (useful for parallelization)
- finish  : the ending index of co-location generation process (useful for parallelization)
- write_per_user : if it is True then the colocation is written to file per user (not all) -- Useful for saving memory when processing data
"""
def generate_colocation(checkins, config, p, k, t_diff, s_diff, start, finish, write_per_user):
  colocations = []
  uids = checkins.keys()
  counter = 0
  skip = 0
  for i in range(start, finish):
    u_i = uids[i]
    df_i = checkins[u_i].sort_values(by=['timestamp'])
    ti_min = checkins[u_i]['timestamp'].min()-t_diff
    ti_max = checkins[u_i]['timestamp'].max()+t_diff
    xi_min = checkins[u_i]['longitude'].min()-s_diff
    xi_max = checkins[u_i]['longitude'].max()+s_diff
    yi_min = checkins[u_i]['latitude'].min()-s_diff
    yi_max = checkins[u_i]['latitude'].max()+s_diff
    # debug('#Checkins of User', u_i, ':', len(df_i))
    si_tree = create_spatial_kd_tree(df_i)
    ti_tree = create_temporal_kd_tree(df_i)
    for j in range(i+1, len(uids)):
      u_j = uids[j]
      df_j = checkins[u_j].sort_values(by=['timestamp'])
      tj_min = checkins[u_j]['timestamp'].min()-t_diff
      tj_max = checkins[u_j]['timestamp'].max()+t_diff
      xj_min = checkins[u_j]['longitude'].min()-s_diff
      xj_max = checkins[u_j]['longitude'].max()+s_diff
      yj_min = checkins[u_j]['latitude'].min()-s_diff
      yj_max = checkins[u_j]['latitude'].max()+s_diff
      ### If there are no intersections between two users' timestamp, then skip
      if ti_max < tj_min or tj_max < ti_min:
        skip += 1
        continue
      ### If the GPS coordinates have no intersections
      if not (xi_min < xj_max and xi_max > xj_min and yi_max > yj_min and yi_min < yj_max ):
        skip += 1
        continue
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
      # if IS_DEBUG is True and j > 10:
      #   break
    counter += 1
    if write_per_user is True:
      write_colocation(colocations, config, p, k, t_diff, s_diff, start, finish)
      del colocations[:]
      _ = gc.collect()
    ### For every N users, shows the progress
    # report_progress(counter, start, finish, context='users', every_n=10)
  debug('Skip', skip, 'user pairs due to the missing time / spatial intersections')
  if write_per_user is True:
    del colocations[:]
    del colocations
    _ = gc.collect()
    return None
  else:
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

def write_colocation(data, config, p, k, t, d, start, finish):
  directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation_part']
  output = io.BytesIO()
  writer = csv.writer(output)
  for row in data:
    writer.writerow(row)
  with open('/'.join([directory, filename.format(p,k,t,d,start,finish)]), 'ab') as f:
    f.write(output.getvalue())

"""
Map and Reduce
"""

def prepare_colocation(config, p, k, t_diff, s_diff, begins, ends):
  ### Clear all intermediate files before doing the map-reduce
  re_format = config['intermediate']['colocation_re']
  working_directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation_part']
  pattern = re.compile(re_format.format(p,k,t_diff,s_diff))
  make_sure_path_exists(working_directory)
  for fname in os.listdir(working_directory):
    if fname.endswith(".csv"):
      if pattern.match(fname):
        remove_file_if_exists('/'.join([working_directory, fname]))
  for i in range(len(begins)):
    with open('/'.join([working_directory, filename.format(p,k,t_diff,s_diff,begins[i],ends[i])]), 'ab') as f:
      f.write(colocation_header)
      # debug('Co-location part %s has been created' % '/'.join([working_directory, filename.format(p,k,t_diff,s_diff,begins[i],ends[i])]))
  debug('Each colocation part file has been created')

def process_map(checkins, config, start, finish, p, k, t_diff=1800, s_diff=0, write_per_user=True):
  ### Execute the mapping process
  debug('Process map [p%d, k%d, t%d, d%d, start%d, finish%d] has started' % (p, k, t_diff, s_diff, start, finish))
  t0 = time.time()
  colocations = generate_colocation(checkins, config, p, k, t_diff, s_diff, start, finish, write_per_user)
  if not write_per_user is True:
    write_colocation(colocations, config, p, k, t_diff, s_diff, start, finish)
  del colocations[:]
  del colocations
  _ = gc.collect()
  elapsed = time.time() - t0
  debug('Process map [p%d, k%d, t%d, d%d, start%d, finish%d] finished in %s seconds' % (p, k, t_diff, s_diff, start, finish, elapsed))

def process_reduce(config, p, k, t_diff, s_diff):
  out_format = config['intermediate']['colocation']
  re_format = config['intermediate']['colocation_re']
  working_directory = config['directory']['colocation']
  # debug(working_directory, out_format, re_format)
  make_sure_path_exists(working_directory)
  pattern = re.compile(re_format.format(p,k,t_diff,s_diff))
  texts = []
  for fname in os.listdir(working_directory):
    if fname.endswith(".csv"):
      if pattern.match(fname):
        with open('/'.join([working_directory, fname]), 'r') as fr:
          for line in fr:
            if line.startswith('user1'):
              continue
            texts.append(line.strip())
  output = '/'.join([working_directory, out_format.format(p,k,t_diff,s_diff)])
  with open(output, 'w') as fw:
    fw.write('%s' % colocation_header)
    for s in texts:
      fw.write('%s\n' % s)
  del texts[:]
  del texts