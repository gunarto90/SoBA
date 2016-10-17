import time
import os
import re
import sys
import getopt
import operator

from joblib import Parallel, delayed

from math import sqrt, pow, exp
from general_utilities import *
from base import *
from classes import *

co_part_filename = 'co_location_gen_p{}_k{}_t{}_d{}_s{}_f{}.csv'
co_raw_part_filename = 'co_raw_gen_p{}_k{}_t{}_d{}_s{}_f{}.csv'
co_raw_filename = 'co_raw_gen_p{}_k{}_t{}_d{}.csv'
co_location_filename = 'co_location_gen_p{}_k{}_t{}_d{}.csv'

def write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder):
    ### Write to file
    texts = []
    texts.append('uid1,uid2,vid,frequency')
    for ss, frequency in co_location.items():
        texts.append('{},{}'.format(ss, frequency))
    filename = working_folder + co_part_filename.format(p, k, t_threshold, d_threshold, i_start, i_finish)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)
    debug('Finished writing co_locations to {}'.format(filename), out_file=False)


"""
    <Next step of each co-location comparison>
    IF User1 has earlier time, then it moves to its next checkins
    ELSE IF User2 has earlier time, then it moves to its next checkins
    ELSE IF both has the same time, then User1 move to its next checkins
"""
def next_co_param(c1, c2, ic1, ic2):
    if c1.time > c2.time:
        ic2 += 1
    else:
        ic1 += 1
    return ic1, ic2

"""
    Time and distance threshold
    Time (in seconds)
    Distance (in meters)
"""
def co_occur(users, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder):
    query_time = time.time()
    co_location = {}
    all_user = []
    for uid, user in users.items():
        all_user.append(uid)
    counter = 0
    texts = []
    texts.append('user1,user2,vid,t_diff,frequency,time1,time2,lat_avg,lon_avg,dist,t_avg')
    # texts.append('user1,user2,lat,lon,t_diff,frequency,time1,time2,t_avg')
    if i_finish == -1:
        i_finish = len(all_user)
    debug('Run co-occurrence from {} to {}'.format(i_start, i_finish), out_file=True)
    for i in range(i_start, i_finish):
        uid1 = all_user[i]
        user1 = users.get(uid1)
        if counter % 1000 == 0:
            debug('{} of {} users ({:.3f}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)), out_file=True, out_stdio=False, callerid='Co-occurrence')
        for j in range(i+1, i_finish):
            uid2 = all_user[j]
            user2 = users.get(uid2)
            if uid1 == uid2:
                continue
            ### No overlapping checkins
            if user1.earliest > user2.latest or user2.earliest > user1.latest:
                continue
            # debug(i,j,len(user1.checkins),len(user2.checkins))
            ic1 = 0
            ic2 = 0
            while ic1 < len(user1.checkins) and ic2 < len(user2.checkins):
                c1 = user1.checkins[ic1]
                c2 = user2.checkins[ic2]
                # debug('[A]:{} ({}), [B]:{} ({})'.format(ic1, len(user1.checkins), ic2, len(user2.checkins)))
                if d_threshold == 0 and c1.vid != c2.vid:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                t_diff = abs(c1.time - c2.time)
                t_avg = (c1.time + c2.time)/2
                lat_avg = (c1.lat + c2.lat)/2
                lon_avg = (c1.lon + c2.lon)/2
                d_diff = haversine(c1.lat, c1.lon, c2.lat, c2.lon)
                if t_diff > t_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                if d_diff > d_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                ss = '{},{},{}'.format(user1.uid, user2.uid, c1.vid)
                texts.append('{},{},{},{},{},{},{},{},{}'.format(user1.uid, user2.uid, c1.vid, t_diff, 1, c1.time, c2.time, lat_avg, lon_avg, d_diff, t_avg))
                co = co_location.get(ss)
                if co is None:
                    co = 0
                co += 1
                co_location[ss] = co
                ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
        counter += 1
    process_time = int(time.time() - query_time)
    debug('Co-occurrence calculation of {0:,} users in {1} seconds'.format((i_finish-i_start), process_time), out_file=True)
    write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder)

    filename = working_folder + co_raw_part_filename.format(p, k, t_threshold, d_threshold, i_start, i_finish)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)

    ### Saving the memory
    del all_user[:]
    del all_user
    del texts[:]
    del texts
    co_location.clear()

"""
Map function
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
n: number of chunks
"""
def mapping(users, p, k, t, d, working_folder, i_start=0, i_finish=-1):
    ### Co-location
    co_occur(users, p, k, t, d, i_start, i_finish, working_folder)

"""
Reduce function
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
"""
def reducing(p, k, t, d, working_folder):
    debug("start reduce processes", out_file=False)
    # pattern = re.compile('(co_location_)(p{}_)(k{}_)(s\d*_)(f\d*_)(t{}_)(d{}).csv'.format(p,k,t,d))
    pattern = re.compile('(co_location_gen_)(p{}_)(k{}_)(t{}_)(d{}_)(s\d*_)(f(-)?\d*).csv'.format(p,k,t,d))
    data = {}
    # dataset, base_folder, working_folder, weekend_folder = init_folder(p)
    # folder = working_folder
    # debug(working_folder)
    ### Extract frequency of meeting
    for file in os.listdir(working_folder):
        if file.endswith(".csv"):
            if pattern.match(file):
                debug(file)
                with open(working_folder + file, 'r') as fr:
                    for line in fr:
                        if line.startswith('uid'):
                            continue
                        line = line.strip()
                        split = line.split(',')
                        if len(split) == 4:
                            _id = '{},{},{}'.format(split[0], split[1], split[2])
                            f = int(split[3])
                            get = data.get(_id)
                            # print(_id, f)
                            if get is None:
                                get = 0
                            f = f + get
                            data[_id] = f
    output = co_location_filename.format(p, k, t, d)
    texts = []
    for _id, f in data.items():
        texts.append('{},{}'.format(_id, f))
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all co location summaries at {}'.format(output), out_file=False)
    del texts[:]
    data.clear()

    ### Extract raw co-occurrence data
    pattern = re.compile('(co_raw_gen_)(p{}_)(k{}_)(t{}_)(d{}_)(s\d*_)(f(-)?\d*).csv'.format(p,k,t,d))
    for file in os.listdir(working_folder):
        if file.endswith(".csv"):
            if pattern.match(file):
                debug(file)
                with open(working_folder + file, 'r') as fr:
                    for line in fr:
                        if line.startswith('uid'):
                            continue
                        texts.append(line.strip())
    output = co_raw_filename.format(p, k, t, d)
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all raw co location data at {}'.format(output), out_file=False)
    del texts[:]
    del texts

# Main function
if __name__ == '__main__':
    ### Brightkite - weekend - 1 core : 600M
    ### Brightkite - weekend - N core @1.4G
    ### Gowalla - weekend - 1 core : 1.4G
    ### Gowalla - weekend - N core @ 2.1G
    ### Brightkite - all - 1 core : 2.1G
    ### Brightkite - all - N core @
    ### Gowalla - all - 1 core : 3.1G
    ### Gowalla - all - N core @
    ### For parallelization
    i_start = 0
    i_finish = -1
    starts = {}
    finish = {}
    starts[0] = [0, 10001, 30001, 55001]
    finish[0] = [10000, 30000, 55000, -1]
    starts[1] = [0, 3001, 8001, 15001, 30001]
    finish[1] = [3000, 8000, 15000, 30000, -1]
    ### Global parameter for the experiments
    ps = []     ### Active project: 0 Gowalla, 1 Brightkite
    ks = []     ### Mode for top k users: 0 Weekend, -1 All users
    ts = []     ### Time threshold
    ds = []     ### Distance threshold
    ### project to be included
    # ps.append(0)
    ps.append(1)
    ### mode to be included
    ks.append(0)
    # ks.append(-1)
    ### time threshold to be included
    HOUR  = 3600
    DAY   = 24 * HOUR
    WEEK  = 7 * DAY
    MONTH = 30 * DAY
    # ts.append(int(0.5 * HOUR))
    # ts.append(1 * HOUR)
    # ts.append(int(1.5 * HOUR))
    # ts.append(2 * HOUR)
    # ts.append(1 * DAY)
    # ts.append(2 * DAY)
    # ts.append(3 * DAY)
    # ts.append(1 * WEEK)
    # ts.append(2 * WEEK)
    # ts.append(1 * MONTH)
    ts.append(2 * MONTH)
    ### distance threshold to be included
    # ds.append(0)
    # ds.append(100)
    # ds.append(250)
    # ds.append(500)
    # ds.append(750)
    ds.append(1000)
    debug("--- Co-occurrence generation started ---")
    for t in ts:
        for d in ds:
            for k in ks:
                for p in ps:
                    debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d))
                    ### Initialize variables
                    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                    dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
                    # ### Initialize dataset
                    users, friends, venues = init(p, k)
                    # ### Sorting users' checkins based on their timestamp, ascending ordering
                    uids = sort_user_checkins(users)
                    ss =starts.get(p)
                    ff = finish.get(p)
                    n_core = 1
                    # n_core = 2
                    # n_core = 3
                    # n_core = 4
                    n_core = len(ss)
                    if n_core == 1:
                        debug('Single core')
                        for i in range(len(ss)):
                            mapping(users, p, k, t, d, working_folder, ss[i], ff[i])
                    else:
                        debug('Number of core: {}'.format(n_core))
                        Parallel(n_jobs=n_core)(delayed(mapping)(users, p, k, t, d, working_folder, ss[i], ff[i]) for i in range(len(ss)))
                    reducing(p, k, t, d, working_folder)
                    ### extracting features
                    # stat_f, stat_d, stat_td, stat_ts = extraction(p, k, t, d, working_folder)
                    # evaluation(friends, stat_f, stat_d, stat_td, stat_ts, p, k, t, d)
                    ### testing extracted csv
                    # testing(p, k, t, d, working_folder)
    debug("--- Co-occurrence generation finished ---")