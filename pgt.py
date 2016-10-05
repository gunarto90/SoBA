from general_utilities import *
from base import *
from classes import *
from math import exp

from colocation import cv_score, sampling, haversine

co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'

### parameters
cd = 1              ### distance parameter in personal density function

def load_user_personal(pd_filename):
    user_p = {}
    with open(pd_filename) as fr:
        for line in fr:
            split = line.strip().split(',')
            uid = int(split[0])
            vid = int(split[1])
            density = float(split[2])
            user_p[(uid, vid)] = density
    return user_p

def user_personal(users, venues, p, k, working_folder, write=True):
    pd_filename = 'pgt_personal_density_p{}_k{}.csv'.format(p, k)
    user_p = {}             ### key: (user_id, loc_id), value: density value (float)
    query_time = time.time()
    counter = 0
    for uid, user in users.items():
        if counter % 100 == 0:
            debug('Processing {} of {} users'.format(counter, len(users.items())))
        for vid, venue in venues.items():
            if venue.count < 2:
                continue
            pi_lock = 0.0   ### density of the location
            for checkin in user.checkins:
                distance = haversine(venue.lat, venue.lon, checkin.lat, checkin.lon)
                pi_lock += exp(-cd * distance)/len(user.checkins)
            if pi_lock > 0.000000001:   ### leave out very small density values to save spaces and computations
                user_p[(uid, vid)] = pi_lock
        counter += 1
    process_time = int(time.time() - query_time)
    debug('Extracting personal density of {0:,} users in {1} seconds'.format(len(users), process_time))
    texts = []
    query_time = time.time()
    for (uid, lid), density in user_p.items():
        texts.append('{},{},{:.9f}'.format(uid, lid, density))
    debug(texts)
    if write is True:
        remove_file_if_exists(working_folder + pd_filename)
        write_to_file_buffered(working_folder + pd_filename, texts)
        process_time = int(time.time() - query_time)
        debug('Writing personal density of {0:,} users in {1} seconds'.format(len(users), process_time))
    return user_p

def venue_global(users, venues, p, k, working_folder, write=True):
    pass

def pgt_personal(user_p, users):
    pass

def pgt_global():
    pass

def pgt_temporal():
    pass

# Main function
if __name__ == '__main__':
    ps = [0]
    ks = [0]
    ts = [3600]
    ds = [0]

    debug('PGT start')
    for p in ps:
        for k in ks:
            dataset, base_folder, working_folder, weekend_folder = init_folder(p)
            users, friends, venues = init(p, k)
            uids = sort_user_checkins(users)
            ### extract personal density values
            user_p = user_personal(users, venues, p, k, working_folder, write=True)
    debug('PGT finished')