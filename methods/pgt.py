#!/usr/bin/env python
import pandas as pd
import numpy as np
### Utility library
import sys
import os
import pickle
from joblib import Parallel, delayed
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import read_config, debug, make_sure_path_exists, is_file_exists, haversine_np, \
    merge_dicts, init_begin_end
from preprocessings.read import extract_friendships, read_colocation_file, determine_social_tie, \
    extract_checkins_per_user, df_uid

### parameters
cd = 1.5    ### distance parameter in personal density function [1,3]
ct = 0.2    ### temporal parameter in temporal dependencies [0.1, 0.3]

def density_location(checkins, grouped, venues, config, start, finish):
    ids = grouped['user'].values
    venues = venues.iloc[0:50]
    for i in range(start, finish):
        u_i = ids[i]
        df = df_uid(checkins, u_i, config, 'user')
        ### To save time, identify the venues that user i could possibly visit 
        ### --> check them from the co-location data (largest -- 0.01)
        ### Otherwise this would take forever
        p_i = venues.apply(lambda x: sum(-1 * cd * haversine_np(df['longitude'], 
            df['latitude'], x['longitude'], x['latitude']))
            , axis=1)

def extract_personal_pgt(config, p, k):
    debug('PGT Personal', 'p', p, 'k', k)
    ### Check if PGT intermediate exists
    dataset_names = config['dataset']
    modes = config['mode']
    pgt_root = config['directory']['pgt']
    make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
    pgt_personal_file = '/'.join([pgt_root, dataset_names[p], \
        config['intermediate']['pgt']['personal'].format(modes[k])])
    if is_file_exists(pgt_personal_file):
        ### Load intermediate file
        # with open(pgt_personal_file, 'rb') as handle:
        #     stat_p = pickle.load(handle)
        pass
    else:
        ### Extract the venue coordinates
        checkins, grouped = extract_checkins_per_user(dataset_names[p], modes[k], config)
        venues = checkins[['location', 'latitude', 'longitude']].drop_duplicates(subset=['location'])
        debug('#Venues', len(venues), 'p', p, 'k', k)
        kwargs = config['kwargs']
        n_core = kwargs['n_core']
        start = kwargs['pgt']['start']
        finish = kwargs['pgt']['finish']
        begins, ends = init_begin_end(n_core, len(grouped), start=start, finish=finish)
        # debug('Begins', begins)
        # debug('Ends', ends)
        Parallel(n_jobs=n_core)(delayed(density_location)(checkins, grouped, venues, config, begins[i], ends[i]) \
            for i in range(len(begins)))
        ### Reduce all parts into an intermediate file

        # with open(pgt_personal_file, 'wb') as handle:
        #     pickle.dump(stat_p, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def extract_personal(config, p, k):
    
#     ### Check if PGT intermediate exists
#     dataset_names = config['dataset']
#     compressed = config['kwargs']['pgt']['compress_output']
#     pgt_root = config['directory']['pgt']
#     make_sure_path_exists('/'.join([pgt_root, dataset_names[p]]))
#     if compressed is True:
#         pgt_name = config['intermediate']['pgt']['evaluation_compressed']
#     else:
#         pgt_name = config['intermediate']['pgt']['evaluation']
#     pgt_name = '/'.join([pgt_root, dataset_names[p], pgt_name.format(p, k, t, d)])
#     if is_file_exists(pgt_name):
#         debug('File %s exists' % pgt_name)
#     else:
#         dataset_names = config['dataset']
#         ### Read (original) friendship from file
#         friend_df = extract_friendships(dataset_names[p], config)
#         colocation_df = read_colocation_file(config, p, k, t, d)
#         ### Find if the two users in the colocated check-ins are friends / stranger
#         colocation_df = determine_social_tie(colocation_df, friend_df)
#         debug('#colocations', len(colocation_df), 'p', p, 'k', k)
#         ### Extract the personal feature
#         personal_df = colocation_df.groupby(['location'])

#         ### Memory management
#         del friend_df
#         del colocation_df
#         del personal_df
#     debug('Finished extract_pgt_features', 'p', p, 'k', k, 't', t, 'd', d)