#!/usr/bin/env python
import pandas as pd
import numpy as np
### Utility library
import sys
import os
import pickle
import time
from joblib import Parallel, delayed
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import read_config, debug, make_sure_path_exists, is_file_exists, haversine_np, \
    merge_dicts, init_begin_end, remove_file_if_exists, entropy
from preprocessings.read import extract_friendships, read_colocation_file, determine_social_tie, \
    extract_checkins_per_user, extract_checkins_per_venue, df_uid

### parameters
cd = 1.5    ### distance parameter in personal density function [1,3]
ct = 0.2    ### temporal parameter in temporal dependencies [0.1, 0.3]

"""
Global feature
- Aggregate the count of (user, location)   #1
- Aggregate the count of (location)         #2
- Divide #1 with #2                         #3
- For each location, calculate the entropy from #3
-- Not tested yet
"""

def calculate_density(row, cd, user, venues):
    temp = venues.loc[venues.location == row['location'], ['longitude', 'latitude']]
    lats = np.full(len(user), temp['latitude'])
    lons = np.full(len(user), temp['longitude'])
    temp = haversine_np(lons, lats, user['longitude'], user['latitude'])
    return sum(np.exp(-cd * temp)/len(user))

def calculate_entropy(row, visits):
    temp = visits.loc[visits.location == row['location']]
    ### Calculate entropy of 
    return entropy(temp['p_x'].values)

def extract_pi_loc_k(checkins, grouped, venues, user_visit, config, p, k, start, finish, feature):
    pgt_root = config['directory']['pgt']
    dataset_names = config['dataset']
    modes = config['mode']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    pgt_file_part = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['%s_part' % feature].format(modes[k], start, finish)])
    if is_file_exists(pgt_file_part) is True:
        pass
    else:
        if feature == 'personal':
            user_visit = user_visit[(user_visit['visit_count'] > 0)]
            ids = grouped['user'].values
            temp_match = venues.isin(user_visit['location'].values)
            temp = venues[temp_match['location']].drop_duplicates()
            debug('#venues in user density calculation', len(temp), 'start', start, 'finish', finish)
        elif feature == 'global':
            visits = checkins[['user', 'location']].drop_duplicates(['user', 'location'])
            visits['p_x'] = visits.groupby(['user', 'location']).transform('count')
            visits['p_y'] = visits.groupby(['location']).transform('count')
            debug(visits.head())
            visits['p_x'] = visits['p_x'] / visits['p_y']
            debug(visits.head())
        result = pd.DataFrame(columns=['user', 'location', 'p_i'])
        t0 = time.time()
        for i in range(start, finish):
            u_i = ids[i]
            if feature == 'personal':
                df = df_uid(checkins, u_i, config, 'user')
                visit_match = user_visit.isin({'location':temp['location'], 'user':df['user'].unique()})
                visit_temp = user_visit[visit_match['location'] | visit_match['user']]
                if len(visit_temp > 0):
                    visit_temp['p_i'] = visit_temp.apply(lambda x: calculate_density(x, cd, df, venues), axis=1)
            elif feature == 'global':
                visit_match = visits.isin({'location':df['location'].unique()})
                visit_temp = visits[visit_match['location']]
                if len(visit_temp > 0):
                    visit_temp['p_i'] = visit_temp.apply(lambda x: calculate_entropy(x, visits), axis=1)
            visit_temp = visit_temp[['user', 'location', 'p_i']]
            result = result.append(visit_temp, ignore_index=True)
        t1 = time.time()
        ### Writing to temp file
        result.drop_duplicates()
        result.to_csv(pgt_file_part, index=False, header=True)
        debug('Finished density calculation into %s in %s seconds' % (pgt_file_part, str(t1-t0)))

def density_location(checkins, grouped, venues, user_visit, config, p, k, start, finish):
    extract_pi_loc_k(checkins, grouped, venues, user_visit, config, p, k, start, finish, 'personal')

def entropy_location(checkins, grouped, venues, user_visit, config, p, k, start, finish):
    extract_pi_loc_k(checkins, grouped, venues, user_visit, config, p, k, start, finish, 'global')

def prepare_extraction(config, feature, p, k):
    ### Check if PGT intermediate exists
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    pgt_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt'][feature].format(modes[k])])
    if is_file_exists(pgt_file):
        debug('PGT %s exists' % feature)
    else:
        if feature == 'personal':
            checkins, grouped = extract_checkins_per_user(dataset_names[p], modes[k], config)
        elif feature == 'global':
            checkins, grouped = extract_checkins_per_venue(dataset_names[p], modes[k], config)
        user_visit_dir = '/'.join([config['directory']['intermediate'], config['dataset'][p]])
        user_visit_name = config['intermediate']['pgt']['user_visit'].format(config['mode'][k])
        final_name = '/'.join([user_visit_dir, user_visit_name])
        ### Using user visit database
        if is_file_exists(final_name):
            user_visit = pd.read_csv(final_name, compression='bz2')
            venues = checkins[['location', 'latitude', 'longitude']].drop_duplicates(subset=['location'])
            debug('#Venues', len(venues), 'p', p, 'k', k)
            kwargs = config['kwargs']
            n_core = kwargs['n_core']
            start = kwargs['pgt'][feature]['start']
            finish = kwargs['pgt'][feature]['finish']
            begins, ends = init_begin_end(n_core, len(grouped), start=start, finish=finish)
            if feature == 'personal':
                function = density_location
            elif feature == 'global':
                function = entropy_location
            ### Map step
            Parallel(n_jobs=n_core)(delayed(function)(checkins, grouped, venues, user_visit, \
                config, p, k, begins[i-1], ends[i-1]) \
                for i in xrange(len(begins), 0, -1))
            ### Reduce step
            result = pd.DataFrame(columns=['user', 'location', 'p_i'])
            for i in range(len(begins)):
                start = begins[i]
                finish = ends[i]
                pgt_file_part = '/'.join([pgt_root, dataset_names[p], \
                    config['intermediate']['pgt']['%s_part' % feature].format(modes[k], start, finish)])
                temp = pd.read_csv(pgt_file_part)
                result = result.append(temp, ignore_index=True)
            result.drop_duplicates(subset=['user', 'location'], inplace=True)
            result.sort_values(['user', 'location'], inplace=True)
            result.to_csv(pgt_file, index=False, header=True)
            ### Clean up mess if needed
            if config['kwargs']['pgt'][feature]['clean_temp'] is True:
                for i in range(len(begins)):
                    start = begins[i]
                    finish = ends[i]
                    pgt_file_part = '/'.join([pgt_root, dataset_names[p], \
                        config['intermediate']['pgt']['%s_part' % feature].format(modes[k], start, finish)])
                    remove_file_if_exists(pgt_file_part)
        else:
            debug('Please generate the user visit first through preprocessing/read.py',
                '(function: generate_user_visit)')

def extract_personal_pgt(config, p, k):
    debug('Extracting PGT Personal', 'p', p, 'k', k)
    prepare_extraction(config, 'personal', p, k)

def extract_global_pgt(config, p, k):
    debug('Extracting PGT Global', 'p', p, 'k', k)
    prepare_extraction(config, 'global', p, k)
