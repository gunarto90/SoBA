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

def user_personal(users, venues, p, k, working_folder, write=True, i_start=0, i_finish=-1):
    debug('Extracting user personal density')
    pd_filename = 'pgt_personal_density_p{}_k{}_s{}_f{}.csv'.format(p, k, i_start, i_finish)
    debug(pd_filename)
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
    debug('Extracting personal density of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time))
    texts = []
    query_time = time.time()
    for (uid, lid), density in user_p.items():
        texts.append('{},{},{:.9f}'.format(uid, lid, density))
    # debug(texts)
    if write is True:
        remove_file_if_exists(working_folder + pd_filename)
        write_to_file_buffered(working_folder + pd_filename, texts)
        process_time = int(time.time() - query_time)
        debug('Writing personal density of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time))
    del all_user[:]
    del all_user
    del texts[:]
    del texts
    return user_p

def venue_global(users, venues, p, k, working_folder, write=True, i_start=0, i_finish=-1):
    debug('Extracting venue global entropy')
    vg_filename = 'pgt_venue_global_p{}_k{}_s{}_f{}.csv'.format(p, k, i_start, i_finish)
    debug(vg_filename)
    venue_list = {}          ### temp for storing probability of visit
    venue_g = {}             ### key: (loc_id), value: entropy value (float)
    counter = 0
    all_user = []
    query_time = time.time()
    for uid1, user in users.items():
        all_user.append(user)
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        user = all_user[i]
        uid = user.uid
        if counter % 10 == 0:
            debug('{} of {} users ({}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)))
        for vid, venue in venues.items():
            pi_lock = 0.0   ### ratio of user visit in the location compared to all population
            u_count = 0     ### count of user visit in the location
            for checkin in user.checkins:
                if checkin.vid == vid:
                    u_count += 1
            if u_count > 0:
                found = venue_list.get(vid)
                if found is None:
                    found = []
                found.append(float(u_count)/venue.count)
                venue_list[vid] = found
        counter += 1
    for vid, list_p in venue_list.items():
        ent  = entropy(list_p)
        freq = len(list_p)
        venue_g[vid] = (ent, freq)
    process_time = int(time.time() - query_time)
    debug('Extracting venue global of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time))
    venue_list.clear()
    texts = []
    query_time = time.time()
    for vid, (ent, freq) in venue_g.items():
        texts.append('{},{:.9f},{}'.format(vid, ent, freq))
    # debug(texts)
    # debug(len(texts))
    # debug(len(venues))
    if write is True:
        remove_file_if_exists(working_folder + vg_filename)
        write_to_file_buffered(working_folder + vg_filename, texts)
        process_time = int(time.time() - query_time)
        debug('Writing personal density of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time))
    del all_user[:]
    del all_user
    del texts[:]
    del texts
    return venue_g

def extraction(p, k, t, d, working_folder):
    fname = co_raw_filename.format(p, k, t, d)
    debug(fname)
    with open(working_folder + fname, 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            # user1,user2,vid,t_diff,frezquency,time1,time2,t_avg
            if line.strip() == 'user1,user2,vid,t_diff,frequency,time1,time2,t_avg':
                continue
            u1 = split[0]
            u2 = split[1]
            vid = split[2]
            t_diff = int(split[3])
            t_avg = float(split[len(split)-1])
            friend = Friend(u1, u2)

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
    i_finish = 100

    CO_DISTANCE = 500
    CO_TIME = 3600
    """
    mode 0: run all factor using files (for personal data and global data)
    mode 1: extract personal data
    mode 2: extract global data
    mode 3: run personal factor evaluation on the co-occurrence
    mode 4: run global factor evaluation on the co-occurrence
    mode 5: run temporal factor evaluation on the co-occurrence
    """
    mode = 2

    debug('PGT start')
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:k:s:f:m:",["project=","topk=","start=","finish=","mode="])
    except getopt.GetoptError:
        err_msg = 'pgt.py -m MODE -p <0 gowalla / 1 brightkite> -k <top k users> -s <start position>'
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
            elif opt in ("-m", "--mode"):
                mode = int(arg)
            elif opt == "--distance":
                CO_DISTANCE = int(arg)
            elif opt == "--time":
                CO_TIME = int(arg)
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

        debug('Co-location time threshold: {}'.format(CO_TIME))
        debug('Co-location distance threshold: {}'.format(CO_DISTANCE))

        users, friends, venues = init(p, k)
        ### Sorting users' checkins based on their timestamp, ascending ordering
        uids = sort_user_checkins(users)
        ### extract personal density values
        if mode == 1:
            user_p = user_personal(users, venues, p, k, working_folder, write=True, i_start=i_start, i_finish=i_finish)
        ### extract global venue entropy
        if mode == 2:
            venue_g = venue_global(users, venues, p, k, working_folder, write=True, i_start=i_start, i_finish=i_finish)
    else:
        ps = [0]
        ks = [0]
        ts = [3600]
        ds = [0]
        if mode == 1 or mode == 2:
            ### Extract required data
            for p in ps:
                for k in ks:
                    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                    users, friends, venues = init(p, k)
                    uids = sort_user_checkins(users)
                    ### extract personal density values
                    if mode == 1:
                        user_p = user_personal(users, venues, p, k, working_folder, write=False, i_start=i_start, i_finish=i_finish)
                    ### extract global venue entropy
                    if mode == 2:
                        venue_g = venue_global(users, venues, p, k, working_folder, write=False, i_start=i_start, i_finish=i_finish)
        else:
            ### Perform PGT calculation
            for p in ps:
                for k in ks:
                    for t in ts:
                        for d in ds:
                            ### extraction
                            extraction(p, k, t, d, working_folder)
                            ### personal
                            ### global
                            ### temporal
                            pass
    debug('PGT finished')