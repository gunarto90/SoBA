import time
import os
import re

from math import radians, cos, sin, asin, sqrt
from general_utilities import *
from base import *
from classes import *

def write_co_location(co_location):
    ### Write to file
    texts = []
    texts.append('uid1,uid2,vid,frequency')
    for ss, frequency in co_location.items():
        texts.append('{},{}'.format(ss, frequency))
    filename = working_folder + 'co_location_p{}_k{}_s{}_f{}_t{}_d{}.csv'.format(ACTIVE_PROJECT, topk, i_start, i_finish, CO_TIME, CO_DISTANCE)
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

def co_occur(users, CO_TIME, CO_DISTANCE):
    """
    Time and distance threshold
    Time (in seconds)
    Distance (in meters)
    """
    t_threshold = CO_TIME
    d_threshold = CO_DISTANCE

    global i_start, i_finish

    query_time = time.time()
    co_location = {}
    all_user = []
    for uid1, user in users.items():
        all_user.append(user)
    count = 0
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
                with open(working_folder + 'last_i_p{}_k{}_s{}_f{}_t{}_d{}.csv'.format(ACTIVE_PROJECT, topk, i_start, i_finish, CO_TIME, CO_DISTANCE), 'w') as fi:
                    fi.write(str(i))
                write_co_location(co_location)
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
                d_diff = haversine(c1.lat, c1.lon, c2.lat, c2.lon)
                if t_diff > t_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                if d_diff > d_threshold:
                    ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
                    continue
                ss = '{},{},{}'.format(user1.uid, user2.uid, c1.vid)
                co = co_location.get(ss)
                if co is None:
                    co = 0
                co += 1
                co_location[ss] = co
                ic1, ic2 = next_co_param(c1, c2, ic1, ic2)
    process_time = int(time.time() - query_time)
    print('Co-occurrence calculation of {0:,} users in {1} seconds'.format(len(users), process_time))
    write_co_location(co_location)

"""
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
n: number of chunks
"""
def map(p, k, d, t, n):
    pass

"""
p: project (gowalla or brightkite)
k: top k (-1 all, 0 weekend, others are top k users)
t: time threshold
d: distance threshold
"""
def reduce(p, k, d, t):
    debug("start reduce processes")
    pattern = re.compile('(co_location_)(p{}_)(k{}_)(s\d*_)(f\d*_)(t{}_)(d{}).csv'.format(p,k,t,d))
    data = {}
    base_folder, working_folder, weekend_folder = init_folder(p)
    folder = working_folder
    # print(folder)
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            if pattern.match(file):
                debug(file)
                with open(folder + file, 'r') as fr:
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
    debug(output)
    texts = []
    for _id, f in data.items():
        texts.append('{},{}'.format(_id, f))
    remove_file_if_exists(folder + output)
    write_to_file_buffered(folder + output, texts)

# Main function
if __name__ == '__main__':
    print("--- Program  started ---")
    # reduce(0, -1, 500, 3600)
    # reduce(0, -1, 0, 3600)
    # reduce(0, 0, 500, 3600)
    print("--- Program finished ---")