from general_utilities import *
from base import *
from classes import *

from math import exp, log
from joblib import Parallel, delayed

import re, os
import sys
import getopt

co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
pd_format = 'pgt_personal_density_p{}_k{}_s{}_f{}.csv'
vg_format = 'pgt_venue_global_p{}_k{}_s{}_f{}.csv'
pd_file = 'pgt_personal_density_p{}_k{}.csv'
vg_file = 'pgt_venue_global_p{}_k{}.csv'

### parameters
cd = 1.5    ### distance parameter in personal density function [1,3]
ct = 0.2    ### temporal parameter in temporal dependencies [0.1, 0.3]

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
    debug('Extracting user personal density p: {}, k: {}'.format(p, k))
    pd_filename = pd_format.format(p, k, i_start, i_finish)
    debug(pd_filename)
    user_p = {}             ### key: (user_id, loc_id), value: density value (float)
    query_time = time.time()
    counter = 0
    skip = 0
    all_user = []
    for uid, user in users.items():
        all_user.append(uid)
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        uid = all_user[i]
        user = users.get(uid)
        if counter % 1000 == 0:
            debug('{} of {} users ({:.3f}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)), callerid='PGT Personal', out_file=True, out_stdio=True)
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
    debug('Extracting personal density of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time), out_file=True)
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
    vg_filename = vg_format.format(p, k, i_start, i_finish)
    debug(vg_filename)
    venue_list = {}          ### temp for storing probability of visit
    venue_g = {}             ### key: (loc_id), value: entropy value (float)
    counter = 0
    all_user = []
    query_time = time.time()
    for uid, user in users.items():
        all_user.append(uid)
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        uid = all_user[i]
        user = users.get(uid)
        if counter % 1000 == 0:
            debug('{} of {} users ({:.3f}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)), callerid='PGT Global', out_file=True, out_stdio=False)
        for vid, venue in venues.items():
            pi_lock = 0.0   ### ratio of user visit in the location compared to all population
            u_count = 0     ### count of user visit in the location
            if venue.count == 0:
                continue
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
    debug('Extracting venue global of {:,} users and {:,} venues in {} seconds'.format(i_finish-i_start, len(venues), process_time), out_file=True)
    debug('Extracted venue global : {}'.format(len(venue_g)))
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
    venue_list.clear()
    del all_user[:]
    del all_user
    del texts[:]
    del texts
    return venue_g

def extraction(working_folder, p, k, t, d, p_density=None, g_entropy=None):
    debug('Extracting PGT', out_file=False)
    fname = co_raw_filename.format(p, k, t, d)
    debug(fname)
    co_f = {}   ### Frequency of co-occurrence
    co_p = {}   ### Max personal weight
    co_g = {}   ### Global
    co_t = {}   ### Temporal
    G0   = {}   ### Frequency function
    G1   = {}   ### Personal average function
    G2   = {}   ### Personal max function
    G3   = {}   ### P + G function
    G4   = {}   ### P + G + T function

    query_time = time.time()
    colocations = {}
    counter = 0
    with open(working_folder + fname, 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            # user1,user2,vid,t_diff,frezquency,time1,time2,t_avg
            if line.strip() == 'user1,user2,vid,t_diff,frequency,time1,time2,t_avg':
                continue
            co = Colocation(split)
            friend = Friend(co.u1, co.u2)
            found = colocations.get(friend)
            if found is None:
                found = []
            found.append(co)
            colocations[friend] = found
            counter += 1
    debug('Found {} co-occurrences'.format(counter))
    # x = colocations.get(Friend(0,7))
    # for ix in x:
    #     debug(ix)
    sort_colocation(colocations)
    # for ix in x:
    #     debug(ix)
    counter = 0
    prev_co = {}
    for friend, cos in colocations.items():
        ### Frequency
        temp_f = co_f.get(friend)
        if temp_f is None:
            temp_f = 0
        ### Personal
        temp_p = []
        ### Global
        temp_g = []
        ### Temporal
        temp_t = []
        ### In all co-occurrences between two users
        for co in cos:
            u1 = co.u1
            u2 = co.u2
            vid = co.vid
            t_diff = co.t_diff
            t_avg = co.t_avg
            ### Frequency
            temp_f += 1
            co_f[friend] = temp_f
            ### Personal
            if p_density is not None:
                w1 = p_density.get((u1, vid))
                w2 = p_density.get((u2, vid))
                if w1 is not None and w2 is not None:
                    wp = -log(w1 * w2)
                    
                else:
                    # debug('Density is not found - user_1: {}, user_2: {}, location:{}'.format(u1, u2, vid))
                    # debug('{},{},{}'.format(u1, u2, vid), clean=True)
                    wp = 0.0
                temp_p.append(wp)
                co_p[co] = wp
            ### Global
            if g_entropy is not None:
                found = g_entropy.get(vid)
                if found is not None:
                    wg = found[0]  # entropy
                else:
                    wg = 0.0
                temp_g.append(wg)
                co_g[co] = wg
            ### Temporal
            found = prev_co.get(friend)
            if found is None:
                wt = 1
            else:
                td = abs(found-t_avg)/3600 ### time difference in hour
                if td <= 1:
                    wt = 1
                else:
                    lt = exp(-ct * td)
                    wt = 1- lt
            temp_t.append(wt)
            co_t[co] = wt
            prev_co[friend] = t_avg
            counter += 1
        ### Function evaluation
        G0[friend] = temp_f
        G1[friend] = sum(temp_p)
        G2[friend] = max(temp_p)
        G3[friend] = max(temp_p)*sum(temp_g)
        total = 0.0
        for i in range(len(temp_g)):
            total += temp_g[i] * temp_t[i]
        G4[friend] = max(temp_p) * total
    process_time = int(time.time() - query_time)
    debug('Extracted {1} co-occurrences in {0} seconds'.format(process_time, counter), out_file=False)
    debug('{}'.format(len(co_p), out_file=False))
    debug('Max p {}'.format(max(co_p.values())))
    debug('Min p {}'.format(min(co_p.values())))
    debug('{}'.format(len(co_g), out_file=False))
    debug('Max g {}'.format(max(co_g.values())))
    debug('Min g {}'.format(min(co_g.values())))
    debug('{}'.format(len(co_t), out_file=False))
    debug('Max t {}'.format(max(co_t.values())))
    debug('Min t {}'.format(min(co_t.values())))
    # debug('{}'.format(len(co_pgt), out_file=False))
    # debug('Max pgt {}'.format(max(co_pgt.values())))
    # debug('Min pgt {}'.format(min(co_pgt.values())))
    return G0, G1, G2, G3

def pgt_personal(working_folder, p, k):
    debug('Extracting personal', out_file=False)
    p_density = {}  ### Density of (uid, loc)
    with open(working_folder + pd_file.format(p,k), 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            uid = int(split[0])
            lid = int(split[1])
            density = float(split[2])
            p_density[(uid, lid)] = density
    debug('Extracted {} personal density'.format(len(p_density)), out_file=False)
    return p_density

def pgt_global(working_folder, p, k):
    debug('Extracting global', out_file=False)
    g_entropy = {}
    with open(working_folder + vg_file.format(p,k), 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            vid = int(split[0])
            entropy = float(split[1])
            frequency = int(split[2])
            entropy = exp(-entropy)
            g_entropy[vid] = (entropy, frequency)
    debug('Extracted {} global entropy'.format(len(g_entropy)), out_file=False)
    return g_entropy

def reduction(working_folder, p, k, mode):
    debug("start reduce processes", out_file=False)
    pattern = re.compile('(pgt_personal_density_)(p{}_)(k{}_)(s\d*_)(f(-)?\d*).csv'.format(p,k))
    texts = []
    for file in os.listdir(working_folder):
        if file.endswith(".csv"):
            if pattern.match(file):
                debug(file)
                with open(working_folder + file, 'r') as fr:
                    for line in fr:
                        texts.append(line.strip())
    if mode == 11:
        output = pd_file.format(p, k)
        remove_file_if_exists(working_folder + output)
        write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all co location summaries at {}'.format(output), out_file=False)
    del texts[:]
    del texts

# Main function
if __name__ == '__main__':
    ### For parallelization
    starts = {}
    finish = {}
    starts[0] = [0, 10001, 30001, 55001]
    finish[0] = [10000, 30000, 55000, -1]
    starts[1] = [0, 3001, 8001, 15001, 30001]
    finish[1] = [3000, 8000, 15000, 30000, -1]
    p = 0
    k = 0
    i_start = 0
    i_finish = -1
    write = True

    CO_DISTANCE = 500
    CO_TIME = 3600
    """
    mode 0: run all factor using files (for personal data and global data)
    mode 1: extract personal data
    mode 2: extract global data
    mode 3: run personal factor evaluation on the co-occurrence
    mode 4: run global factor evaluation on the co-occurrence
    mode 11: run reduction personal
    """
    mode = 0

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
        elif mode == 2:
            venue_g = venue_global(users, venues, p, k, working_folder, write=True, i_start=i_start, i_finish=i_finish)
        ### reducing personal density
        elif mode == 11:
            reduce(working_folder, p, k, mode)
    else:
        modes = [0]
        ps = [1]
        ks = [0]
        ts = [3600]
        ds = [0]
        for mode in modes:
            debug('Mode: {}'.format(mode), out_file=False)
            if mode == 1 or mode == 2:
                ### Extract required data
                for p in ps:
                    for k in ks:
                        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                        users, friends, venues = init(p, k)
                        uids = sort_user_checkins(users)
                        ### Parallelization
                        ss = starts.get(p)
                        ff = finish.get(p)
                        # n_core = 1
                        # n_core = 2
                        # n_core = 3
                        # n_core = 4
                        n_core = len(ss)
                        # debug('Number of core: {}'.format(n_core))
                        ### extract personal density values
                        if mode == 1:
                            if n_core == 1:
                                user_p = user_personal(users, venues, p, k, working_folder, write=False, i_start=i_start, i_finish=i_finish)
                            else:
                                user_p = Parallel(n_jobs=n_core)(delayed(user_personal)(users, venues, p, k, working_folder, write=write, i_start=ss[i], i_finish=ff[i]) for i in range(len(ss)))
                                pass
                        ### extract global venue entropy
                        if mode == 2:
                            venue_g = venue_global(users, venues, p, k, working_folder, write=True, i_start=i_start, i_finish=i_finish)
            ### reducing personal density
            elif mode == 11:
                for p in ps:
                    for k in ks:
                        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                        reduction(working_folder, p, k, mode)
            else:
                ### Perform PGT calculation
                for p in ps:
                    for k in ks:
                        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                        p_density = None
                        g_entropy = None
                        ### personal
                        if mode == 3 or mode == 0:
                            p_density = pgt_personal(working_folder, p, k)
                        ### global
                        if mode == 4 or mode == 0:
                            g_entropy = pgt_global(working_folder, p, k)
                        for t in ts:
                            for d in ds:
                                ### extraction
                                extraction(working_folder, p, k, t, d, p_density, g_entropy)
                                pass
    debug('PGT finished')