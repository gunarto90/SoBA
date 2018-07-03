#!/usr/bin/env python
"""
Please run this code under the root directory using the command 'python preprocessing/read.py'
"""
### General library
import numpy as np
import pandas as pd
### Utility library
import pickle
import sys
import os.path
import gc
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local library
from common.functions import IS_DEBUG, debug, fn_timer, make_sure_path_exists, is_file_exists, read_config, \
  remove_file_if_exists
from common.visual import gmplot

RAW_CHECKIN_FILE = 'checkin.csv'

### Global
config_file = 'config.json'
final_column = ['user', 'timestamp', 'latitude', 'longitude', 'location']

"""
Generate three standardized compressed csv files which consist of all, weekend, and weekday data
"""
@fn_timer
def generate_results(root, dataset, df):
  ### Standardized
  checkin_all = 'checkin_all.csv.gz'
  checkin_weekday = 'checkin_weekday.csv.gz'
  checkin_weekend = 'checkin_weekend.csv.gz'
  ### Writing results to files
  df.to_csv('/'.join([root, dataset, checkin_all]), header=True, index=False, compression ='gzip')
  df_weekday = df[df.index.dayofweek < 5]
  df_weekday.to_csv('/'.join([root, dataset, checkin_weekday]), header=True, index=False, compression ='gzip')
  df_weekend = df[df.index.dayofweek >= 5]
  df_weekend.to_csv('/'.join([root, dataset, checkin_weekend]), header=True, index=False, compression ='gzip')

"""
Reading the foursquare checkin dataset
"""
@fn_timer
def read_foursquare2012_checkin(root, write=True):
  dataset = 'foursquare'
  debug('Read Checkin %s' %dataset)
  df = pd.read_csv('/'.join([root, dataset, RAW_CHECKIN_FILE]), parse_dates=['time'])
  debug(df.describe(include='all'))
  debug(df.head())
  ### Create a UNIX timestamp column from the datetime format
  df['timestamp'] = df['time'].values.astype(np.int64) // 10 ** 9
  ### Set the datetime as the index
  df = df.set_index('time')
  ### Reordering columns
  df = df[final_column]
  ### Error checking
  # odd = df.loc[df.longitude>-80, ['longitude', 'latitude']]
  ### Writing results to files
  if write is True:
    generate_results(root, dataset, df)

"""
Reading the checkin dataset from SNAP Stanford
dataset = 'gowalla' or 'brightkite'
"""
@fn_timer
def read_snap_stanford_checkin(root, dataset='gowalla', write=True):
  debug('Read SNAP Stanford Checkin %s' %dataset)
  df = pd.read_csv('/'.join([root, dataset, RAW_CHECKIN_FILE]), header=None, names=['user','timestamp','latitude','longitude','location'])
  debug(df.describe(include='all'))
  debug(df.head())
  ### Create a datetime column as the index
  df['time'] = pd.to_datetime(df['timestamp'], unit='s')
  df = df.set_index('time')
  debug(df.head())
  ### Reordering columns
  df = df[final_column]
  ### Writing results to files
  if write is True:
    generate_results(root, dataset, df)

"""
Read the standardized data
"""
def read_processed(root, dataset='gowalla', mode='all', id='user', filter=True):
  filename = 'checkin_{}.csv.gz'.format(mode)
  'user', 'timestamp', 'latitude', 'longitude', 'location'
  dtypes = {
      'user':np.int_,'timestamp':np.int_,
      'latitude':np.float_,'longitude':np.float_,
      'location':np.int_
    }
  df = pd.read_csv('/'.join([root, dataset, filename]), dtype=dtypes)
  df['u_count'] = df.groupby('user')['user'].transform('count')
  df['v_count'] = df.groupby('location')['location'].transform('count')
  df['visit_unique'] = df.groupby('user')['location'].transform('nunique')
  ### Apply filtering
  ### User count > 10 and location visit > 1 (otherwise there is no co-location) 
  ### and must have visited 10 different places
  if filter is True:
    df = df[(df['u_count'] > 10) & (df['v_count'] > 1) & (df['visit_unique'] > 10)]
  df.drop(['u_count', 'v_count', 'visit_unique'], axis=1, inplace=True)
  ### Adding spatiotemporal information from dataframe
  if id == 'checkin':
    grouped = None
  else:
    aggregations = {
      'timestamp' : {
        't_avg':'mean', 
        't_min':'min', 
        't_max':'max'
      },
      'latitude' : {
        'lat_avg':'mean', 
        'lat_min':'min', 
        'lat_max':'max'
      },
      'longitude' : {
        'lon_avg':'mean', 
        'lon_min':'min', 
        'lon_max':'max'
      }
    }
    grouped = df.groupby([id]).agg(aggregations)
    grouped.columns = grouped.columns.droplevel(level=0)
    grouped.reset_index(inplace=True)
    grouped['t_avg'] = (grouped['t_avg']).apply(np.int64)
    round = {
      't_avg':0, 
      't_min':0, 
      't_max':0,
      'lat_avg':2, 
      'lat_min':2, 
      'lat_max':2,
      'lon_avg':2, 
      'lon_min':2, 
      'lon_max':2
    }
    grouped = grouped.round(round)
    grouped.sort_values(by=['lon_min', 'lat_min'], inplace=True)
    grouped = grouped[[id, 't_avg', 'lat_avg', 'lon_avg', 't_min', 't_max', 
      'lat_min', 'lat_max', 'lon_min', 'lon_max']]
  return df, grouped

def preprocess_data(root):
  write = False
  read_foursquare2012_checkin(root, write)
  read_snap_stanford_checkin(root, 'brightkite', write)
  read_snap_stanford_checkin(root, 'gowalla', write)

def visualize_data(df):
  test_limit = 100
  temp = df[0:test_limit]  ### For testing purpose --> to speed-up and understand the data
  gmplot(temp)

@fn_timer
def extract_checkins(dataset_name, mode, config, id, filter):
  debug('Processing %s [%s] for each %s [filter=%s]' % (dataset_name, mode, id, filter))
  dataset_root = config['directory']['dataset']
  df, grouped = read_processed(dataset_root, dataset_name, mode, id, filter)
  debug('#checkins', len(df))
  if grouped is not None:
    debug('#%ss' % id, len(grouped))
  return df, grouped

"""
Extract all the checkins and group them on each user
Input:
- dataset_name (foursquare, gowalla, brightkite)
- mode (all, weekday, weekend)
- config: config.json filename
Output:
- Single dataframe consists of all checkins
- Grouped dataframe based on the user
"""
def extract_checkins_per_user(dataset_name, mode, config, filter=True):
  df, grouped = extract_checkins(dataset_name, mode, config, 'user', filter=filter)
  return df, grouped

"""
Extract all the checkins and group them on each user
Input:
- dataset_name (foursquare, gowalla, brightkite)
- mode (all, weekday, weekend)
- config: config.json filename
Output:
- Single dataframe consists of all checkins
- Grouped dataframe based on the venue
"""
def extract_checkins_per_venue(dataset_name, mode, config, filter=True):
  df, grouped = extract_checkins(dataset_name, mode, config, 'location', filter=filter)
  return df, grouped

"""
Extract all the checkins and group them on each user
Input:
- dataset_name (foursquare, gowalla, brightkite)
- mode (all, weekday, weekend)
- config: config.json filename
Output:
- Single dataframe consists of all checkins
- None
"""
def extract_checkins_all(dataset_name, mode, config, filter=True):
  df, grouped = extract_checkins(dataset_name, mode, config, 'checkin', filter=filter)
  return df, grouped

def extract_friendships(dataset_name, config):
  dataset_root = config['directory']['dataset']
  friendship_name = '/'.join([dataset_root, dataset_name, 'friend.csv'])
  colocation_dtypes = {
        'user1':np.int64,'user2':np.int64
    }
  friend_df = pd.read_csv(friendship_name, dtype=colocation_dtypes)
  return friend_df

"""
Retrieving the check-ins of user 'uid'
"""
def df_uid(df, uid, config, force_id=None):
  if force_id is None:
    id = config['kwargs']['colocation']['run_by']
  else:
    id = force_id
  return df.loc[df[id] == uid]

@fn_timer
def read_colocation_file(config, p, k, t, d, chunksize=None, usecols=None):
    ### Read co-location from file
    is_read_compressed = config['kwargs']['read_compressed']
    colocation_root = config['directory']['colocation']
    if is_read_compressed is False:
        colocation_name = config['intermediate']['colocation']['csv']
    else:
        colocation_name = config['intermediate']['colocation']['compressed']
    colocation_fullname = '/'.join([colocation_root, colocation_name.format(p, k, t, d)])
    colocation_dtypes = {
        'user1':np.int_,'user2':np.int_,
        'location1':np.int_,'location2':np.int_,
        'time1':np.int_,'time2':np.int_,
        'lat1':np.float_,'lon1':np.float_,'lat2':np.float_,'lon2':np.float_,
        't_diff':np.int_,'s_diff':np.float_
    }
    if chunksize is None:
      colocation_df = pd.read_csv(colocation_fullname, dtype=colocation_dtypes, usecols=usecols)
    else:
      colocation_df = pd.read_csv(colocation_fullname, dtype=colocation_dtypes, chunksize=chunksize, usecols=usecols)
    return colocation_df

def read_colocation_chunk(config, p, k, t, d, usecols=None):
  chunksize = 10 ** 6
  return read_colocation_file(config, p, k, t, d, chunksize, usecols)

def determine_social_tie(colocation_df, friend_df):
    colocation_df = pd.merge(colocation_df, friend_df, on=['user1','user2'], how='left', indicator='link')
    colocation_df['link'] = np.where(colocation_df.link == 'both', 1, 0)
    return colocation_df

def generate_user_visit(config):
  kwargs = config['kwargs']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  for dataset_name in datasets:
    p = config['dataset'].index(dataset_name)
    for mode in modes:
      k = config['mode'].index(mode)
      out_dir = '/'.join([config['directory']['intermediate'], config['dataset'][p]])
      out_name = config['intermediate']['pgt']['user_visit'].format(config['mode'][k])
      final_name = '/'.join([out_dir, out_name])
      if is_file_exists(final_name):
        debug('File %s already exists' % final_name)
      else:
        df, _ = extract_checkins_all(dataset_name, mode, config, filter=True)
        visits = df.groupby(['user', 'location'])['timestamp'].count().reset_index()
        visits.rename(columns={"timestamp": "visit_count"}, inplace=True)
        u_count = df.groupby('user')['timestamp'].count().reset_index()
        u_count.rename(columns={"timestamp": "user_count"}, inplace=True)
        v_count = df.groupby('location')['timestamp'].count().reset_index()
        v_count.rename(columns={"timestamp": "location_count"}, inplace=True)
        visits = visits.join(u_count, on='user', how='outer', rsuffix='r')
        visits = visits.join(v_count, on='location', how='outer', rsuffix='r')
        visits = visits[['user', 'location', 'visit_count', 'user_count', 'location_count']]
        visits.fillna(0, inplace=True)
        ### All of these must have the same amount
        debug('Total #Checkins', len(df))
        debug('#Total user visits', int(visits['visit_count'].sum()))
        debug('#Total user counts', int(visits.drop_duplicates(['user'])['user_count'].sum()))
        debug('#Total location counts', int(visits.drop_duplicates(['location'])['location_count'].sum()))
        visits.to_csv(final_name, header=True, index=False, compression='bz2')
        del visits, df
        gc.collect()

def sort_colocation(config):
  kwargs = config['kwargs']
  datasets = kwargs['active_dataset']
  modes = kwargs['active_mode']
  t_diffs = kwargs['ts']
  s_diffs = kwargs['ds']
  is_read_compressed = config['kwargs']['read_compressed']
  colocation_root = config['directory']['colocation']
  if is_read_compressed is False:
      colocation_name = config['intermediate']['colocation']['csv']
      compression = None
  else:
      colocation_name = config['intermediate']['colocation']['compressed']
      compression = 'bz2'
  for dataset_name in datasets:
    p = config['dataset'].index(dataset_name)
    for mode in modes:
      k = config['mode'].index(mode)
      for t in t_diffs:
        for d in s_diffs:
          colocation_df = read_colocation_file(config, p, k, t, d)
          colocation_df.sort_values(['user1', 'user2', 'time1', 'time2', 'location1', 'location2'], inplace=True)
          colocation_fullname = '/'.join([colocation_root, colocation_name.format(p, k, t, d)])
          remove_file_if_exists(colocation_fullname)
          colocation_df.to_csv(colocation_fullname, index=False, header=True, compression=compression)
          debug('Finished sorting %s' % colocation_fullname)

@fn_timer
def main():
  ### Read config
  config = read_config()
  kwargs = config['kwargs']

  ### Read original data and generate standardized data
  if kwargs['preprocessing']['run_extraction'] is True:
    if kwargs['preprocessing']['read_original'] is True:
      dataset_root = config['directory']['dataset']
      preprocess_data(dataset_root)
  ### Extract user visit from co-location
  if kwargs['preprocessing']['user_visit'] is True:
    generate_user_visit(config)
  ### Sorting co-location based on several criteria
  if kwargs['preprocessing']['sort_colocation'] is True:
    sort_colocation(config)

if __name__ == '__main__':
  main()