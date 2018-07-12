#!/usr/bin/env python
import pandas as pd
import numpy as np
### Utility library
import sys
import os
import pickle
import time
import gc
import math
from joblib import Parallel, delayed
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import read_config, debug, make_sure_path_exists, is_file_exists, haversine_np, \
    merge_dicts, init_begin_end, remove_file_if_exists, entropy_np, applyParallel
from preprocessings.read import extract_friendships, read_colocation_file, determine_social_tie, \
    extract_checkins_per_user, extract_checkins_per_venue, df_uid

### parameters
cd = 1.5    ### distance parameter in personal density function [1,3]
ct = 0.2    ### temporal parameter in temporal dependencies [0.1, 0.3]

def calculate_density(row, cd, user, venues):
    temp = venues.loc[venues.location == row['location'], ['longitude', 'latitude']]
    lats = np.full(len(user), temp['latitude'])
    lons = np.full(len(user), temp['longitude'])
    temp = haversine_np(lons, lats, user['longitude'], user['latitude'])
    return sum(math.exp(-cd * temp)/len(user))

def calculate_entropy(arr):
    if sum(arr) > 0:
        return entropy_np(arr)
    else: return 0.0

def calculate_personal(row, density):
    density_i = density.loc[(density.user == row['user1']) & (density.location == row['location1'])]
    density_j = density.loc[(density.user == row['user2']) & (density.location == row['location2'])]
    if row['user1'] == row['user2']: return 0.0
    if density_i.empty is True:
        pi = 0.0
    else:
        pi = density_i['p_i'].values[0]
    if density_j.empty is True:
        pj = 0.0
    else:
        pj = density_j['p_i'].values[0]
    if pi == 0.0 or pj == 0.0:
        return 0.0
    return -math.log(pi*pj)

def calculate_global(row, entropy):
    entropy_i = entropy.loc[(entropy.location) == row['location1']]
    entropy_j = entropy.loc[(entropy.location) == row['location2']]
    if entropy_i.empty is True:
        pi = 0.0
    else:
        pi = entropy_i['p_i'].values[0]
    if entropy_j.empty is True:
        pj = 0.0
    else:
        pj = entropy_j['p_i'].values[0]
    if pi != 0.0 and pj != 0.0:
        return (pi+pj)/2    ### Average of two entropy location
    else:
        return pi + pj ### Because at least one of them is 0.0

def extract_pi_loc_k(checkins, grouped, venues, user_visit, config, p, k, start, finish, feature):
    pgt_part_root = config['directory']['pgt_temp']
    dataset_names = config['dataset']
    modes = config['mode']
    make_sure_path_exists('/'.join([pgt_part_root, dataset_names[p]]))
    pgt_file_part = '/'.join([pgt_part_root, dataset_names[p], \
        config['intermediate']['pgt']['%s_part' % feature].format(modes[k], start, finish)])
    if is_file_exists(pgt_file_part) is True:
        pass
    else:
        user_visit = user_visit[(user_visit['visit_count'] > 0)]
        #debug('start', start, 'finish', finish)
        if feature == 'personal':
            ids = grouped['user'].values
            grouping = 'user'
            result = pd.DataFrame(columns=['user', 'location', 'p_i'])
        elif feature == 'global':
            ids = grouped['location'].values
            grouping = 'location'
            result = pd.DataFrame(columns=['location', 'p_i'])
        t0 = time.time()
        for i in range(start, finish):
            u_i = ids[i]
            df = df_uid(checkins, u_i, config, grouping)
            visit_match = user_visit.isin({grouping:df[grouping].unique()})
            visit_temp = user_visit[visit_match[grouping]]
            if len(visit_temp > 0):
                if feature == 'personal':
                    ### Extract the p_i of each user's visit
                    visit_temp['p_i'] = visit_temp.apply(lambda x: calculate_density(x, cd, df, venues), axis=1)
                    visit_temp = visit_temp[['user', 'location', 'p_i']]
                    result = result.append(visit_temp, ignore_index=True)
                elif feature == 'global':
                    ### Aggregate visit on each location
                    aggregations = {
                        'user_count':{
                            'entropy':lambda x: calculate_entropy(x)
                        },
                    }
                    grouped = visit_temp.groupby(['location']) \
                        .agg(aggregations)
                    grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
                    grouped.rename(columns={"user_count_entropy": "p_i"}, inplace=True)
                    grouped.reset_index(inplace=True)
                    # debug(grouped.columns.values)
                    # debug(grouped.head())
                    grouped = grouped[['location', 'p_i']]
                    result = result.append(grouped, ignore_index=True)
        t1 = time.time()
        ### Writing to temp file
        if feature == 'personal':
            result.drop_duplicates(subset=['user', 'location'], inplace=True)
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
    pgt_part_root = config['directory']['pgt_temp']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    pgt_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt'][feature].format(modes[k])])
    if is_file_exists(pgt_file) is True:
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
        if is_file_exists(final_name) is True:
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
            if feature == 'personal':
                result = pd.DataFrame(columns=['user', 'location', 'p_i'])
            elif feature == 'global':
                result = pd.DataFrame(columns=['location', 'p_i'])
            for i in range(len(begins)):
                start = begins[i]
                finish = ends[i]
                pgt_file_part = '/'.join([pgt_part_root, dataset_names[p], \
                    config['intermediate']['pgt']['%s_part' % feature].format(modes[k], start, finish)])
                temp = pd.read_csv(pgt_file_part)
                result = result.append(temp, ignore_index=True)
            if feature == 'personal':
                result.drop_duplicates(subset=['user', 'location'], inplace=True)
                result.sort_values(['user', 'location'], inplace=True)
            elif feature == 'global':
                result.sort_values(['location'], inplace=True)
            debug('#User Visits', len(result))
            result.to_csv(pgt_file, index=False, header=True)
            ### Clean up mess if needed
            if config['kwargs']['pgt'][feature]['clean_temp'] is True:
                for i in range(len(begins)):
                    start = begins[i]
                    finish = ends[i]
                    pgt_file_part = '/'.join([pgt_part_root, dataset_names[p], \
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

def transform_colocation_pgt(config, p, k, t, d, feature):
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    pgt_part_root = config['directory']['pgt_temp']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    make_sure_path_exists('/'.join([pgt_part_root, dataset_names[p]]))
    pgt_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt'][feature].format(modes[k])])
    ### Check if PGT intermediate exists
    if is_file_exists(pgt_file) is False:
        debug('PGT %s does not exists' % feature)
        debug('Please run PGT %s factor extraction first' % feature)
        return None
    else:
        g0 = '/'.join([pgt_root, dataset_names[p], \
            config['intermediate']['pgt']['pgt_g0_%s' % feature].format(modes[k], t, d)])
        if is_file_exists(g0) is False:
            if feature == 'personal':
                ### columns=['user', 'location', 'p_i']
                personal_density = pd.read_csv(pgt_file)
                col_name = 'wp'
            elif feature == 'global':
                ### columns=['location', 'p_i']
                entropy_location = pd.read_csv(pgt_file)
                col_name = 'wg'
            ### user1,user2,location1,location2,time1,time2,lat1,lon1,lat2,lon2,t_diff,s_diff
            ### Evaluate the weight for each colocation
            ### Map step
            i = 0
            chunksize=10 ** 5
            debug('chunksize for transform_colocation_pgt', chunksize)
            for colocation_df in read_colocation_file(config, p, k, t, d, chunksize=chunksize, usecols=['user1', 'user2', 'location1', 'location2']):
                g0_part = '/'.join([pgt_part_root, dataset_names[p], \
                    config['intermediate']['pgt']['pgt_g0_%s_part' % feature].format(modes[k], t, d, i)])
                debug('Processing', feature, 'part', g0_part)
                if is_file_exists(g0_part) is False:
                    if feature == 'personal':
                        colocation_df[col_name] = colocation_df.apply(lambda x: calculate_personal(x, personal_density), axis=1)
                    elif feature == 'global':
                        colocation_df[col_name] = colocation_df.apply(lambda x: calculate_global(x, entropy_location), axis=1)
                    colocation_df.to_csv(g0_part, index=False, header=True, compression='bz2')
                i += 1
            ### Reduce step
            colocation_df = pd.DataFrame(columns=['user1', 'user2', 'location1', 'location2', col_name])
            condition = True
            i = 0
            while(condition is True):
                ### Iterate over all chunks
                g0_part = '/'.join([pgt_part_root, dataset_names[p], \
                    config['intermediate']['pgt']['pgt_g0_%s_part' % feature].format(modes[k], t, d, i)])
                if is_file_exists(g0_part) is False:
                    condition = False
                    break
                temp = pd.read_csv(g0_part)
                colocation_df = colocation_df.append(temp, ignore_index=True)
                i += 1
            if config['kwargs']['pgt'][feature]['clean_temp'] is True:
                condition = True
                i = 0
                while(condition is True):
                    g0_part = '/'.join([pgt_part_root, dataset_names[p], \
                        config['intermediate']['pgt']['pgt_g0_%s_part' % feature].format(modes[k], t, d, i)])
                    if is_file_exists(g0_part) is False:
                        condition = False
                        break
                    remove_file_if_exists(g0_part)
                    i += 1
            colocation_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            colocation_df.fillna(0, inplace=True)
            colocation_df.to_csv(g0, index=False, header=True, compression='bz2')
            gc.collect()
        else:
            colocation_df = pd.read_csv(g0)
            debug('Loaded g0 %s successfully [%s]' % (feature, g0))
        return colocation_df

def personal_factor(config, p, k, t, d):
    g1 = None
    g2 = None
    ### Intermediate file -- check if exists
    pgt_root = config['directory']['pgt']
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    g1_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['pgt_g1'].format(modes[k], t, d)])
    g2_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['pgt_g2'].format(modes[k], t, d)])
    if is_file_exists(g1_file) is True and is_file_exists(g2_file) is True:
        g1 = pd.read_csv(g1_file)
        g2 = pd.read_csv(g2_file)
    else:
        ### If it does not exist
        feature = 'personal'
        colocation_df = transform_colocation_pgt(config, p, k, t, d, feature)
        ### Aggregate the weight for each user pair
        g1 = colocation_df.groupby(['user1', 'user2'])['wp'].agg(['mean', 'count'])
        g2 = colocation_df.groupby(['user1', 'user2'])['wp'].agg(['max', 'count'])
        g1.reset_index(inplace=True)
        g2.reset_index(inplace=True)
        g1.sort_values(['user1', 'user2'], inplace=True)
        g2.sort_values(['user1', 'user2'], inplace=True)
        g1['g1'] = g1['mean'] * g1['count']
        g2['g2'] = g2['max'] * g2['count']
        g1['g1'] = g1['g1']/max(g1['g1'])
        g2['g2'] = g2['g2']/max(g2['g2'])
        g1.to_csv(g1_file, header=True, index=False, compression='bz2')
        g2.to_csv(g2_file, header=True, index=False, compression='bz2')
        # debug(g1.head())
        # debug(g2.head())
        del colocation_df
    return g1, g2

def global_factor(config, p, k, t, d, g2):
    g3 = None
     ### Intermediate file -- check if exists
    pgt_root = config['directory']['pgt']
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    g3_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['pgt_g3'].format(modes[k], t, d)])
    if is_file_exists(g3_file) is True:
        g3 = pd.read_csv(g3_file)
    else:
        feature = 'global'
        colocation_df = transform_colocation_pgt(config, p, k, t, d, feature)
        g3 = colocation_df.groupby(['user1', 'user2'])['wg'].agg(['sum'])
        g3.reset_index(inplace=True)
        g3.sort_values(['user1', 'user2'], inplace=True)
        g3['g3'] = g2['g2'] * g3['sum']
        g3['g3'] = g3['g3']/max(g3['g3'])
        g3.to_csv(g3_file, header=True, index=False, compression='bz2')
        del colocation_df
    return g3

def lambda_temporal(group):
    df = group[['user1', 'user2']]
    t_diff = group['time'].diff().fillna(0).values
    lt = 1-np.exp(-ct * t_diff)
    g4 = group['wg'] * lt
    df['wt'] = g4.values
    return df

def temporal_factor(config, p, k, t, d, g2):
    g4 = None
     ### Intermediate file -- check if exists
    pgt_root = config['directory']['pgt']
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    g4_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['pgt_g4'].format(modes[k], t, d)])
    if is_file_exists(g4_file) is True:
        g4 = pd.read_csv(g4_file)
    else:
        global_df = transform_colocation_pgt(config, p, k, t, d, 'global')
        colocation_df = read_colocation_file(config, p, k, t, d, \
            usecols=['user1', 'user2', 'time1', 'time2'])
        colocation_df['wg'] = global_df['wg'].values
        colocation_df['time'] = (colocation_df['time1'] + colocation_df['time2'])/2
        colocation_df.drop(columns=['time1', 'time2'], inplace=True)
        groups = colocation_df.groupby(['user1', 'user2'])
        g4 = applyParallel(config, groups, lambda_temporal)
        g4 = g4.groupby(['user1', 'user2'])['wt'].agg(['sum'])
        g4.reset_index(inplace=True)
        g4.sort_values(['user1', 'user2'], inplace=True)
        g4['g4'] = g2['g2'] * g4['sum']
        g4[g4 < 0] = 0.0  ### Prevent negatives in values
        g4['g4'] = g4['g4']/max(g4['g4'])
        g4.to_csv(g4_file, header=True, index=False, compression='bz2')
        del colocation_df, groups
    return g4

def extract_pgt(config, p, k, t, d):
    dataset_names = config['dataset']
    compressed = config['kwargs']['compress_output']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    if compressed is True:
        pgt_name = config['intermediate']['pgt']['pgt_output_compressed']
        compression = 'bz2'
    else:
        pgt_name = config['intermediate']['pgt']['pgt_output']
        compression = None
    intermediate_file = '/'.join([pgt_root, dataset_names[p], pgt_name.format(p, k, t, d)])
    if is_file_exists(intermediate_file) is False:
        ### Extracting each feature
        if config['kwargs']['pgt']['extract_pgt']['run'] is True:
            g1 = None
            g2 = None
            g3 = None
            g4 = None
            if config['kwargs']['pgt']['extract_pgt']['personal'] is True:
                g1, g2 = personal_factor(config, p, k, t, d)        ### P in PGT
                debug('Finished loading personal factor', 'p', p, 'k', k, 't', t, 'd', d)
            if config['kwargs']['pgt']['extract_pgt']['global'] is True:
                g3 = global_factor(config, p, k, t, d, g2)          ### PG in PGT
                debug('Finished loading global factor', 'p', p, 'k', k, 't', t, 'd', d)
            if config['kwargs']['pgt']['extract_pgt']['temporal'] is True:
                g4 = temporal_factor(config, p, k, t, d, g2)        ### PGT in PGT
                debug('Finished loading temporal factor', 'p', p, 'k', k, 't', t, 'd', d)
            ### Merging all together
            if config['kwargs']['pgt']['extract_pgt']['merge'] is True:
                if g1 is not None and g2 is not None and g3 is not None and g4 is not None:
                    df = g1[['user1', 'user2', 'g1']].merge(g2[['user1', 'user2', 'g2']], on=['user1', 'user2'])
                    df = df.merge(g3[['user1', 'user2', 'g3']], on=['user1', 'user2'])
                    df = df.merge(g4[['user1', 'user2', 'g4']], on=['user1', 'user2'])
                    friend_df = extract_friendships(dataset_names[p], config)
                    df = determine_social_tie(df, friend_df)
                    df.to_csv(intermediate_file, header=True, index=False, compression=compression)