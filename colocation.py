import time
import os
import re
import sys
import getopt

from math import radians, cos, sin, asin, sqrt
from general_utilities import *
from base import *
from classes import *

def write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder):
    ### Write to file
    texts = []
    texts.append('uid1,uid2,vid,frequency')
    for ss, frequency in co_location.items():
        texts.append('{},{}'.format(ss, frequency))
    filename = working_folder + 'co_location_p{}_k{}_t{}_d{}_s{}_f{}.csv'.format(p, k, t_threshold, d_threshold, i_start, i_finish)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    distance = km * 1000
    return distance

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
def co_occur(users, p, k, t_threshold, d_threshold, BACKUP, i_start, i_finish, working_folder):
    query_time = time.time()
    co_location = {}
    all_user = []
    for uid1, user in users.items():
        all_user.append(user)
    count = 0
    texts = []
    texts.append('user1,user2,vid,t_diff,frequency,time1,time2,t_avg')
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        user1 = all_user[i]
        # print('{} of {} users ({}%) [{}]'.format(i, len(all_user), float(i)*100/len(all_user), datetime.now()))
        if i % 10 == 0:
            debug('{} of {} users ({}%) [{}]'.format(i, len(all_user), float(i)*100/len(all_user), datetime.now()))
        if BACKUP > 0:
            if i > i_start and i % BACKUP == 0:
                ### Save current progress
                with open(working_folder + 'last_i_p{}_k{}_t{}_d{}_s{}_f{}.csv'.format(p, k, t_threshold, d_threshold, i_start, i_finish, ), 'w') as fi:
                    fi.write(str(i))
                write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder)
        for j in range(i+1, i_finish):
            user2 = all_user[j]
            if user1.uid == user2.uid:
                continue
            ### No overlapping checkins
            if user1.earliest > user2.latest or user2.earliest > user1.latest:
                continue
            count += 1
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
                d_diff = haversine(c1.lat, c1.lon, c2.lat, c2.lon)
                if t_diff > t_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                if d_diff > d_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                ss = '{},{},{}'.format(user1.uid, user2.uid, c1.vid)
                texts.append('{},{},{},{},{},{},{},{}'.format(user1.uid, user2.uid, c1.vid, t_diff, 1, c1.time, c2.time, t_avg))
                co = co_location.get(ss)
                if co is None:
                    co = 0
                co += 1
                co_location[ss] = co
                ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
    process_time = int(time.time() - query_time)
    print('Co-occurrence calculation of {0:,} users in {1} seconds'.format(len(users), process_time))
    write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder)

    filename = working_folder + 'co_raw_p{}_k{}_t{}_d{}_s{}_f{}.csv'.format(p, k, t_threshold, d_threshold, i_start, i_finish)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)

"""
Map function
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
n: number of chunks
"""
def mapping(p, k, d, t, BACKUP, working_folder, i_start=0, i_finish=-1):
    ### Initialize dataset
    users, friends, venues = init(p, k)
    ### Sorting users' checkins based on their timestamp, ascending ordering
    uids = sort_user_checkins(users)
    ### Co-location
    co_occur(users, p, k, t, d, BACKUP, i_start, i_finish, working_folder)

"""
Reduce function
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
"""
def reducing(p, k, d, t, working_folder):
    debug("start reduce processes")
    # pattern = re.compile('(co_location_)(p{}_)(k{}_)(s\d*_)(f\d*_)(t{}_)(d{}).csv'.format(p,k,t,d))
    pattern = re.compile('(co_location_)(p{}_)(k{}_)(t{}_)(d{}_)(s\d*_)(f\d*).csv'.format(p,k,t,d))
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
    output = 'co_location_p{}_k{}_t{}_d{}.csv'.format(p, k, t, d)
    texts = []
    for _id, f in data.items():
        texts.append('{},{}'.format(_id, f))
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all co location summaries at {}'.format(output))

    ### Extract raw co-occurrence data
    texts.clear()
    pattern = re.compile('(co_raw_)(p{}_)(k{}_)(t{}_)(d{}_)(s\d*_)(f\d*).csv'.format(p,k,t,d))
    for file in os.listdir(working_folder):
        if file.endswith(".csv"):
            if pattern.match(file):
                debug(file)
                with open(working_folder + file, 'r') as fr:
                    for line in fr:
                        if line.startswith('uid'):
                            continue
                        texts.append(line.strip())
    output = 'co_raw_p{}_k{}_t{}_d{}.csv'.format(p, k, t, d)
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all raw co location data at {}'.format(output))

def extraction(p, k, d, t, working_folder):
    pass

def evaluation(p, k, d, t, working_folder):
    pass

def test(p, k, d, t, i):
    print(p, k, d, t, i)
    time.sleep(1)

# Main function
if __name__ == '__main__':
    ACTIVE_PROJECT = 0
    topk = 0
    i_start = 0
    i_finish = -1
    BACKUP = 1000
    CO_DISTANCE = 500
    CO_TIME = 3600  
    print("--- Program  started ---")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:k:s:f:",["project=","topk=","start=","finish","backup=","distance=","time="])
    except getopt.GetoptError:
        err_msg = 'colocation.py -p <0 gowalla / 1 brightkite> -k <top k users> -s <start position> [optional] --backup=<every #users to backup> --distance=<co-location distance threshold> --time=<co-location time threshold>'
        debug(err_msg, 'opt error')
        sys.exit(2)
    if len(opts) > 0:
        for opt, arg in opts:
            if opt == '-h': 
                debug(err_msg, 'opt error')
                sys.exit()
            elif opt in ("-p", "--project"):
                ACTIVE_PROJECT = int(arg)
            elif opt in ("-k", "--topk"):
                topk = int(arg)
            elif opt in ("-s", "--start"):
                i_start = int(arg)
            elif opt in ("-f", "--finish"):
                i_finish = int(arg)
            elif opt == "--backup":
                BACKUP = int(arg)
            elif opt == "--distance":
                CO_DISTANCE = int(arg)
            elif opt == "--time":
                CO_TIME = int(arg)

        dataset, base_folder, working_folder, weekend_folder = init_folder(ACTIVE_PROJECT)

        debug('Selected project: {}'.format(dataset[ACTIVE_PROJECT]))
        debug('Starting position: {}'.format(i_start))
        debug('Finishing position: {}'.format(i_finish))
        debug('Backup every {} users'.format(BACKUP))
        debug('Co-location time threshold: {}'.format(CO_TIME))
        debug('Co-location distance threshold: {}'.format(CO_DISTANCE))
        if topk > 0:
            debug('Top {} users are selected'.format(topk))
        elif topk == 0:
            debug('Evaluating weekend checkins')
        elif topk == -1:
            debug('Evaluating all checkins')
        mapping(ACTIVE_PROJECT, topk, CO_DISTANCE, CO_TIME, BACKUP, working_folder, i_start, i_finish)
        # reducing(ACTIVE_PROJECT, topk, CO_DISTANCE, CO_TIME, working_folder)
        # evaluation(ACTIVE_PROJECT, topk, CO_DISTANCE, CO_TIME)
    else:
        ps = [0]
        ks = [0]
        ts = [3600]
        ds = [0]

        for p in ps:
            for k in ks:
                for d in ds:
                    for t in ts:
                        print('p:{}, k:{}, d:{}, t:{}'.format(p, k, d, t))
                        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                        # mapping(p, k, d, t, BACKUP, working_folder)
                        reducing(p, k, d, t, working_folder)
                        # evaluation(p, k, d, t)
    print("--- Program finished ---")