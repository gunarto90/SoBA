from general_utilities import *
from base import *
from classes import *
from math import exp
import sys
import getopt

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

def user_personal(users, venues, p, working_folder, write=True, i_start=0, i_finish=-1):
    pd_filename = 'pgt_personal_density_p{}_s{}_f{}.csv'.format(p, i_start, i_finish)
    user_p = {}             ### key: (user_id, loc_id), value: density value (float)
    query_time = time.time()
    counter = 0
    skip = 0
    all_user = []
    for uid1, user in users.items():
        all_user.append(user)
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        user = all_user[i]
        uid = user.uid
        if counter % 10 == 0:
            debug('{} of {} users ({}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)))
            # debug('Skipped {} unused venues'.format(skip))
        for vid, venue in venues.items():
            if venue.count < 2:
                skip += 1
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
    # debug(texts)
    if write is True:
        remove_file_if_exists(working_folder + pd_filename)
        write_to_file_buffered(working_folder + pd_filename, texts)
        process_time = int(time.time() - query_time)
        debug('Writing personal density of {0:,} users in {1} seconds'.format(len(users), process_time))
    return user_p

def venue_global(users, venues, p, working_folder, write=True):
    pass

def pgt_personal(user_p, users):
    pass

def pgt_global():
    pass

def pgt_temporal():
    pass

# Main function
if __name__ == '__main__':
    p = 0
    k = 0
    i_start = 0
    i_finish = -1

    debug('PGT start')
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:k:s:f:",["project=","topk=","start=","finish"])
    except getopt.GetoptError:
        err_msg = 'colocation.py -p <0 gowalla / 1 brightkite> -k <top k users> -s <start position> [optional] --backup=<every #users to backup> --distance=<co-location distance threshold> --time=<co-location time threshold>'
        debug(err_msg, 'opt error')
        sys.exit(2)
    if len(opts) > 0:
        for opt, arg in opts:
            if opt in ("-p", "--project"):
                p = int(arg)
            elif opt in ("-k", "--topk"):
                k = int(arg)
            elif opt in ("-s", "--start"):
                i_start = int(arg)
            elif opt in ("-f", "--finish"):
                i_finish = int(arg)
        ### Initialize dataset
        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
        debug('Selected project: {}'.format(dataset[p]))
        if k > 0:
            debug('Top {} users are selected'.format(k))
        elif k == 0:
            debug('Evaluating weekend checkins')
        elif k == -1:
            debug('Evaluating all checkins')
        debug('Starting position: {}'.format(i_start))
        debug('Finishing position: {}'.format(i_finish))
        users, friends, venues = init(p, k)
        ### Sorting users' checkins based on their timestamp, ascending ordering
        uids = sort_user_checkins(users)
        ### extract personal density values
        user_p = user_personal(users, venues, p, working_folder, write=True, i_start=i_start, i_finish=i_finish)
    else:
        ps = [0]
        ks = [0]
        for p in ps:
            for k in ks:
                dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                users, friends, venues = init(p, k)
                uids = sort_user_checkins(users)
                ### extract personal density values
                user_p = user_personal(users, venues, p, working_folder, write=True)
                ### extract global mobility entropy

    debug('PGT finished')