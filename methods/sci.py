#!/usr/bin/env python
import pandas as pd
import numpy as np
### Utility library
import math
import pickle
import sys
import os
import warnings
warnings.filterwarnings("ignore")   ### Disable warnings
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import IS_DEBUG, read_config, debug, fn_timer, entropy, make_sure_path_exists, is_file_exists
from preprocessings.read import extract_friendships

"""
Private functions
"""
@fn_timer
def extract_visit_per_venue(checkins):
    visit_per_venue = {}    ### Total visit in a venue (by all users)
    p_l  = []               ### #Visit of user U to each location
    for df in checkins.values():
        temp = df.groupby('location')['location'].size().reset_index(name='counts')
        for row in temp.itertuples():
            current_count = visit_per_venue.get(row.location)
            if current_count is None:
                current_count = 0
            visit_per_venue[row.location] = row.counts + current_count
        temp.set_index('location', inplace=True)
        p_l.append(temp.to_dict('index'))
        del temp
    return visit_per_venue, p_l

@fn_timer
def extract_aggregated_visit(visit_per_venue, p_l):
    p_ul = {}   ### #Visit of all users in a venue (aggregated and normalized to range[0,1])
    for vid, frequency in visit_per_venue.items():
        p_ul[vid] = []
        for arr in p_l:
            if arr.get(vid) is not None:
                p_ul[vid].append(float(arr[vid]['counts'])/frequency)
    return p_ul

@fn_timer
def read_colocation_file(config, p, k, t, d):
    ### Read co-location from file
    is_read_compressed = config['sci']['read_compressed']
    colocation_root = config['directory']['colocation']
    if is_read_compressed is False:
        colocation_name = config['intermediate']['colocation']
    else:
        colocation_name = config['intermediate']['colocation_compressed']
    colocation_fullname = '/'.join([colocation_root, colocation_name.format(p, k, t, d)])
    colocation_dtypes = {
        'user1':np.int64,'user2':np.int64,
        'location1':np.int64,'location2':np.int64,
        'time1':np.int64,'time2':np.int64,
        'lat1':np.float64,'lon1':np.float64,'lat2':np.float64,'lon2':np.float64,
        't_diff':np.int64,'s_diff':np.float64
    }
    colocation_df = pd.read_csv(colocation_fullname, dtype=colocation_dtypes)
    return colocation_df

def determine_social_tie(colocation_df, friend_df):
    colocation_df = pd.merge(colocation_df, friend_df, on=['user1','user2'], how='left', indicator='link')
    colocation_df['link'] = np.where(colocation_df.link == 'both', 1, 0)
    return colocation_df

@fn_timer
def write_statistics(df, config, p, k, t, d):
    dataset_names = config['dataset']
    compressed = config['kwargs']['sci']['compress_output']
    sci_root = config['directory']['sci']
    make_sure_path_exists('/'.join([sci_root, dataset_names[p]]))
    if compressed is True:
        sci_name = config['intermediate']['sci']['evaluation_compressed']
        compression = 'bz2'
    else:
        sci_name = config['intermediate']['sci']['evaluation']
        compression = None
    df.to_csv('/'.join([sci_root, dataset_names[p], sci_name.format(p, k, t, d)]), \
        header=True, index=False, compression=compression)

def calculate_diversity(arr):
    if len(arr) == 1:
        return 0    ### No diversity if only 1 co-location
    stat = {}
    for x in arr:
        temp = stat.get(x)
        if temp is None:
            temp = 0
        temp += 1
        stat[x] = temp
    ent = entropy(stat.values())
    return ent

def calculate_duration(arr):
    if len(arr) == 1:
        return 0    ### No duration if only 1 co-location
    return max(arr)-min(arr)

def calculate_popularity(arr, stat_lp):
    temp = []
    for x in arr:
        pop = stat_lp.get(x)
        if pop is not None:
            temp.append(pop)
    if len(temp) > 0:
        return sum(temp)/len(temp)
    else:
        return 0

def calculate_sigma(arr, miu):
    result = 0.0
    max_time = max(arr)
    for i in arr:
        result += math.pow((i/max_time)-miu,2)
        # result += math.pow(i/len(arr)-miu,2)
    return result

def calculate_stability(groups):
    results = []
    for name, df in groups:
        if len(df) <= 1:
            results.append(0.0) ### No stability over 1 co-location
        else:
            ### Calculate the stability
            df['time_avg'] = (df['time1']+df['time2'])/2
            ### Average meeting time between each co-location
            df['diff'] = df['time_avg'].diff()
            miu_xy = (df['diff'].sum(skipna=True)/len(df))
            max_timediff = df['diff'].max(skipna=True)
            ### Average standard deviation of co-location time
            sigma_xy = calculate_sigma(df['time_avg'].values, miu_xy)            
            ### Density of each co-location
            rho_xy = math.sqrt(sigma_xy/len(df))
            ### Final weight of the stability feature
            w_s = math.exp(-(miu_xy+rho_xy)/max_timediff)
            results.append(w_s)
    return np.array(results)

"""
Public functions
"""

"""
Extract the popularity for each venues
inputs:
- checkins  : checkins made by each user (dict:{int, dataframe})
- config    : configuration file
- p         : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k         : mode for dataset (0: all, 1: weekday, 2: weekend)
output:
- stat_lp   : location popularity of each venue (dict:{int, float})
"""
def extract_popularity(checkins, config, p, k):
    intermediate_root = config['directory']['intermediate']
    dataset_names = config['dataset']
    modes = config['mode']
    popularity_intermediate_file = config['intermediate']['sci']['popularity']
    pickle_directory = '/'.join([intermediate_root, dataset_names[p]])
    make_sure_path_exists(pickle_directory)
    pickle_filename = '/'.join([pickle_directory, popularity_intermediate_file.format(modes[k])])
    if not is_file_exists(pickle_filename):
        stat_lp = {}  ### Popularity score for location l
        visit_per_venue, p_l = extract_visit_per_venue(checkins)
        p_ul = extract_aggregated_visit(visit_per_venue, p_l)
        ### Evaluate the weight for each venue
        for vid, arr in p_ul.items():
            if len(arr) > 0:
                ent = entropy(arr)
                stat_lp[vid] = ent
            else:
                stat_lp[vid] = 0.0
        ### Memory management
        del p_l[:]
        del p_l
        visit_per_venue.clear()
        p_ul.clear()
        del visit_per_venue, p_ul
        ### Write to pickle intermediate file
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(stat_lp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_filename, 'rb') as handle:
            stat_lp = pickle.load(handle)
    ### Return the result
    return stat_lp

"""
Extract all the co-location's features
inputs:
- stat_lp   : location popularity of each venue (dict:{int, float})
- p         : dataset (0: gowalla, 1: brightkite, 2: foursquare)
- k         : mode for dataset (0: all, 1: weekday, 2: weekend)
- t         : time threshold for co-location criterion (in seconds)
- d         : spatial threshold for co-location criterion (in meters -- after running "extract_partial_colocation.py)
"""
def extract_colocation_features(stat_lp, config, p, k, t, d):
    dataset_names = config['dataset']
    ### Read (original) friendship from file
    friend_df = extract_friendships(dataset_names[p], config)
    colocation_df = read_colocation_file(config, p, k, t, d)
    ### Extract the user pairs
    user_pairs = colocation_df[['user1', 'user2']].drop_duplicates()
    ### Find if the two users in the colocated check-ins are friends / stranger
    colocation_df = determine_social_tie(colocation_df, friend_df)
    ### Find the stability value for each co-location pairs
    groups = colocation_df.groupby(['user1', 'user2', 'link'])
    stability = calculate_stability(groups)
    ### Extracting the basic statistic from the co-location dataset
    aggregations = {
        'lat1':'count',                         ### Frequency
        'location1':{
            'diversity':calculate_diversity,    ### Diversity
            'popularity':lambda x: calculate_popularity(x, stat_lp)
        },
        'location2':{
            'diversity':calculate_diversity,    ### Diversity
            'popularity':lambda x: calculate_popularity(x, stat_lp)
        },
        'time1':{
            'duration':calculate_duration       ### Duration
        },
        'time2':{
            'duration':calculate_duration       ### Duration
        }
    }
    grouped = groups.agg(aggregations)
    grouped['stability'] = stability
    ### Fix the naming schemes of column names
    grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
    ### Applying Normalization
    normalized = ['time1_duration', 'time2_duration', 
        'location1_diversity', 'location2_diversity',
        'location1_popularity', 'location2_popularity',
        "stability_"
    ]
    for column in normalized:
        grouped[column] = grouped[column]/max(grouped[column])
    ### Applying average
    grouped['time1_duration'] = (grouped['time1_duration'] + grouped['time2_duration'])/2
    grouped['location1_diversity'] = (grouped['location1_diversity'] + grouped['location2_diversity'])/2
    grouped['location1_popularity'] = (grouped['location1_popularity'] + grouped['location2_popularity'])/2
    ### Removing unecessary columns
    grouped.drop(['time2_duration'], axis=1, inplace=True)
    grouped.drop(['location2_diversity'], axis=1, inplace=True)
    grouped.drop(['location2_popularity'], axis=1, inplace=True)
    ### Renaming columns
    grouped.reset_index(inplace=True)
    grouped.rename(columns={"lat1_count": "frequency", 
        "location1_diversity": "diversity", 
        "time1_duration":"duration",
        "location1_popularity": "popularity",
        "stability_":"stability",
        "user1":"uid1", "user2":"uid2"
        }, inplace=True)
    ### Reordering the columns
    grouped = grouped[['uid1', 'uid2', 'frequency', 'diversity', 'duration', 'stability', 'popularity', 'link']]
    debug(grouped.columns.values)
    ### Removing all co-location less than two co-occurrences
    grouped = grouped[(grouped['frequency'] > 1)]
    ### Write the result into a csv output
    write_statistics(grouped, config, p, k, t, d)

    ### Memory management
    del friend_df
    del colocation_df
    del user_pairs
    del grouped