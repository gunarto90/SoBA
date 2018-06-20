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
import shutil
### KD Tree
from scipy import spatial
sys.setrecursionlimit(10000)  ### To make sure recursion limit do not stop the KD-tree
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from preprocessings.read import df_uid
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

"""
Co-location generation
Input:
- checkins: dataframe of user checkins (user, timestamp, latitude, longitude, location)
- grouped : dataframe of aggregated statistics (<id>, t_avg, t_min, t_max, lat_avg, lat_min, lat_max, lon_avg, lon_min, lon_max)
- config  : configuration file
- p       : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k       : mode for dataset (0: all, 1: weekday, 2: weekend)
- t_diff  : time threshold for co-location criterion (in seconds)
- s_diff  : spatial threshold for co-location criterion (in lat/lon degree) (1 degree = 111 Km)
- start   : the beginning index of co-location generation process (useful for parallelization)
- finish  : the ending index of co-location generation process (useful for parallelization)
- write_per_instance : if it is True then the colocation is written to file per user (not all) -- Useful for saving memory when processing data
"""
def generate_colocation(checkins, grouped, config, p, k, t_diff, s_diff, start, finish, write_per_instance):
  colocations = []
  run_by = config['kwargs']['colocation']['run_by']
  ids = grouped[run_by].values
  counter = 0
  total_skip = 0
  is_debugging_colocation = config['kwargs']['colocation']['debug']
  skip_tolerance = config['kwargs']['colocation']['early_stop']
  for i in range(start, finish):
    consecutive_skip = 0
    if i < 0 or i >len(ids):
      break
    u_i = ids[i]
    df_i = df_uid(checkins, u_i, config)
    stats_i = df_uid(grouped, u_i, config)
    si_tree = create_spatial_kd_tree(df_i)
    ti_tree = create_temporal_kd_tree(df_i)
    for j in range(i+1, len(ids)):
      u_j = ids[j]
      stats_j = df_uid(grouped, u_j, config)
      ### If there are no intersections between two users' timestamp, then skip
      if stats_i['t_max'].values[0]+t_diff < stats_j['t_min'].values[0] \
        or stats_j['t_max'].values[0]+t_diff < stats_i['t_min'].values[0]:
        total_skip += 1
        consecutive_skip += 1
        # debug(i, j, 'consecutive_skip', consecutive_skip, 
        #   stats_i['lat_min'].values[0], stats_j['lat_min'].values[0], \
        #   stats_i['lat_max'].values[0], stats_j['lat_max'].values[0], \
        #   stats_i['lon_min'].values[0], stats_j['lon_min'].values[0], \
        #   stats_i['lon_max'].values[0], stats_j['lon_max'].values[0], \
        # )
        del u_j, stats_j
        if consecutive_skip > skip_tolerance and skip_tolerance > 0:
          # debug('early stop time')
          total_skip += len(ids)-j-1
          break
        else:
          continue
      df_j = df_uid(checkins, u_j, config)
      ### If the GPS coordinates have no intersections
      if( stats_i['lat_min'].values[0] > stats_j['lat_max'].values[0]+s_diff or \
          stats_i['lat_max'].values[0]+s_diff < stats_j['lat_min'].values[0] or \
          stats_i['lon_min'].values[0] > stats_j['lon_max'].values[0]+s_diff or \
          stats_i['lon_max'].values[0]+s_diff < stats_j['lon_min'].values[0] 
      ):
        total_skip += 1
        consecutive_skip += 1
        # debug(i, j, 'consecutive_skip', consecutive_skip, 
        #   stats_i['lat_min'].values[0], stats_j['lat_min'].values[0], \
        #   stats_i['lat_max'].values[0], stats_j['lat_max'].values[0], \
        #   stats_i['lon_min'].values[0], stats_j['lon_min'].values[0], \
        #   stats_i['lon_max'].values[0], stats_j['lon_max'].values[0], \
        # )
        del df_j, u_j, stats_j
        if consecutive_skip > skip_tolerance and skip_tolerance > 0:
          # debug('early stop spatial')
          total_skip += len(ids)-j-1
          break
        else:
          continue
      else:
        consecutive_skip = 0
      tj_tree = create_temporal_kd_tree(df_j)
      ### temporal co-occurrence
      t_idx = ti_tree.query_ball_tree(tj_tree, t_diff)
      t_count = sum(len(x) for x in t_idx)
      if t_count > 0:
        ### spatial co-occurrence
        sj_tree = create_spatial_kd_tree(df_j)
        s_idx = si_tree.query_ball_tree(sj_tree, s_diff)
        s_count = sum(len(x) for x in s_idx)
        ### Only if both temporal and spatial co-occurrence > 0
        if s_count > 0:
          ### Finding the intersection and adding colocations to the list
          result = extract_radius_search_results(df_i, df_j, s_idx, t_idx)
          if result is not None and len(result)>0:
            colocations.extend(result)
            del result[:]
            del result
        del s_idx, sj_tree
      del tj_tree, t_idx, df_j, u_j, stats_j
      ### For testing purpose
      if is_debugging_colocation is True and j > i+11:
        break
    ### Prepare for the next iteration
    counter += 1
    if write_per_instance is True:
      if colocations is not None:
        if len(colocations)>0:
          write_colocation(colocations, config, p, k, t_diff, s_diff, start, finish)
        del colocations[:]
    ### Clear-up memory
    del u_i, df_i, si_tree, ti_tree, stats_i
    _ = gc.collect()
  del ids
  debug('Skipped', total_skip, 'user pairs due to the missing time / spatial intersections')
  if write_per_instance is True:
    ### Delete the last colocations set if it is per-user
    if colocations is not None:
      del colocations[:]
      del colocations
      _ = gc.collect()
    return None
  else:
    return colocations

"""
Co-location report generation (version 1)
df_i    : Dataframe of user i
df_j    : Dataframe of user j
s_idx   : List of list of all locations within the "radius"
t_idx   : List of list of all locations within the "time difference"
"""
def extract_radius_search_results(df_i, df_j, s_idx, t_idx):
  results = []
  if len(s_idx) != len(t_idx):
    return None
  for i in range(len(s_idx)):
    temp = set(s_idx[i]).intersection(t_idx[i])
    for j in temp:
      di = df_i.iloc[i].round(3)
      dj = df_j.iloc[j].round(3)
      ### Format: user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff
      t_diff = int(math.fabs(di['timestamp'] - dj['timestamp']))
      s_diff = round(haversine(di['latitude'], di['longitude'], dj['latitude'], dj['longitude']), 2)
      obj = (int(di['user']), int(dj['user']), int(di['location']), int(dj['location']), di['timestamp'],dj['timestamp'],
        di['latitude'], di['longitude'], dj['latitude'], dj['longitude'],
        t_diff, s_diff
      )
      results.append(obj)
  return results

"""
Write colocation results into file
- data    : list of colocations
- config  : configuration file
- p       : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k       : mode for dataset (0: all, 1: weekday, 2: weekend)
- t       : time threshold for co-location criterion (in seconds)
- d       : spatial threshold for co-location criterion (in lat/lon degree) (1 degree = 111 Km)
- start   : the beginning index of co-location generation process (useful for parallelization)
- finish  : the ending index of co-location generation process (useful for parallelization)
"""
def write_colocation(data, config, p, k, t, d, start, finish):
  if data is None or len(data) < 1:
    return
  directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation_part']
  dataset_name = config['dataset'][p]
  output = io.BytesIO()
  writer = csv.writer(output)
  for row in data:
    writer.writerow(row)
  with open('/'.join([directory, dataset_name, filename.format(p,k,t,d,start,finish)]), 'ab') as f:
    f.write(output.getvalue())

"""
Map and Reduce functions
"""

"""
Prepare the files for the co-location generation (Make sure file and directory exist)
"""
def prepare_colocation(config, p, k, t_diff, s_diff, begins, ends):
  working_directory = config['directory']['colocation']
  filename  = config['intermediate']['colocation_part']
  dataset_name = config['dataset'][p]
  make_sure_path_exists('/'.join([working_directory, dataset_name]))
  ### Prepare the files
  for i in range(len(begins)):
    with open('/'.join([working_directory, dataset_name, filename.format(p,k,t_diff,s_diff,begins[i],ends[i])]), 'wb'):
      pass
  debug('Each colocation part file has been created')

"""
Process map in the map-reduce scheme (Generating co-location list per chunk)
- checkins: dataframe of user checkins (user, timestamp, latitude, longitude, location)
- grouped : dataframe of aggregated statistics (<id>, t_avg, t_min, t_max, lat_avg, lat_min, lat_max, lon_avg, lon_min, lon_max)
- config  : configuration file
- p       : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k       : mode for dataset (0: all, 1: weekday, 2: weekend)
- t_diff  : time threshold for co-location criterion (in seconds)
- s_diff  : spatial threshold for co-location criterion (in lat/lon degree) (1 degree = 111 Km)
- write_per_instance : if it is True then the colocation is written to file per user (not all) -- Useful for saving memory when processing data
"""
def process_map(checkins, grouped, config, start, finish, p, k, t_diff=1800, s_diff=0, write_per_instance=True):
  ### Execute the mapping process
  debug('Process map [p%d, k%d, t%d, d%.3f, start%d, finish%d] has started' % (p, k, t_diff, s_diff, start, finish))
  t0 = time.time()
  colocations = generate_colocation(checkins, grouped, config, p, k, t_diff, s_diff, start, finish, write_per_instance)
  if write_per_instance is False:
    write_colocation(colocations, config, p, k, t_diff, s_diff, start, finish)
    if colocations is not None:
      del colocations[:]
      del colocations
      _ = gc.collect()
  elapsed = time.time() - t0
  debug('Process map [p%d, k%d, t%d, d%.3f, start%d, finish%d] finished in %s seconds' % (p, k, t_diff, s_diff, start, finish, elapsed))

"""
Process reduce in the map-reduce scheme (Combining all files)
- config  : configuration file
- p       : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k       : mode for dataset (0: all, 1: weekday, 2: weekend)
- t_diff  : time threshold for co-location criterion (in seconds)
- s_diff  : spatial threshold for co-location criterion (in lat/lon degree) (1 degree = 111 Km)
"""
def process_reduce(config, p, k, t_diff, s_diff):
  out_format = config['intermediate']['colocation']
  re_format = config['intermediate']['colocation_re']
  working_directory = config['directory']['colocation']
  dataset_name = config['dataset'][p]
  # debug(working_directory, out_format, re_format)
  make_sure_path_exists('/'.join([working_directory, dataset_name]))
  pattern = re.compile(re_format.format(p,k,t_diff,s_diff))
  file_list = []
  for fname in os.listdir('/'.join([working_directory, dataset_name])):
    if fname.endswith(".csv"):
      if pattern.match(fname):
        file_list.append('/'.join([working_directory, dataset_name, fname]))
  output = '/'.join([working_directory, dataset_name, out_format.format(p,k,t_diff,s_diff)])
  with open(output, 'wb') as fw:
    fw.write('%s' % colocation_header)
  with open(output,'ab') as wfd:
    for f in file_list:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)
            #10MB per writing chunk to avoid reading big file into memory.