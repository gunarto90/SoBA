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

last_backup_filename = 'last_i_p{}_k{}_t{}_d{}_s{}_f{}.csv'
co_part_filename = 'co_location_p{}_k{}_t{}_d{}_s{}_f{}.csv'
co_raw_part_filename = 'co_raw_p{}_k{}_t{}_d{}_s{}_f{}.csv'
co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
co_location_filename = 'co_location_p{}_k{}_t{}_d{}.csv'
evaluation_filename = 'evaluation_p{}_k{}_t{}_d{}.csv'

def write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder):
    ### Write to file
    texts = []
    texts.append('uid1,uid2,vid,frequency')
    for ss, frequency in co_location.items():
        texts.append('{},{}'.format(ss, frequency))
    filename = working_folder + co_part_filename.format(p, k, t_threshold, d_threshold, i_start, i_finish)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)

def write_evaluation(summaries, p, k, t, d):
    texts = []
    texts.append('uid1,uid2,frequency,diversity,duration,stability,link')
    for friend, evaluation in summaries.items():
        texts.append(str(evaluation))
    filename = working_folder + evaluation_filename.format(p, k, t, d)
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, texts)
    debug('Finished writing to {}'.format(filename))

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
    for uid1, user in users.items():
        all_user.append(user)
    counter = 0
    texts = []
    texts.append('user1,user2,vid,t_diff,frequency,time1,time2,t_avg')
    # texts.append('user1,user2,lat,lon,t_diff,frequency,time1,time2,t_avg')
    if i_finish == -1:
        i_finish = len(all_user)
    for i in range(i_start, i_finish):
        user1 = all_user[i]
        if i % 100 == 0:
            debug('{} of {} users ({}%)'.format(i, i_finish, float(counter)*100/(i_finish-i_start)))
        for j in range(i+1, i_finish):
            user2 = all_user[j]
            if user1.uid == user2.uid:
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
                # lat_avg = (c1.lat + c2.lat)/2
                # lon_avg = (c1.lon + c2.lon)/2
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
        counter += 1
    process_time = int(time.time() - query_time)
    print('Co-occurrence calculation of {0:,} users in {1} seconds'.format(len(users), process_time))
    write_co_location(co_location, p, k, t_threshold, d_threshold, i_start, i_finish, working_folder)

    filename = working_folder + co_raw_part_filename.format(p, k, t_threshold, d_threshold, i_start, i_finish)
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
    output = co_location_filename.format(p, k, t, d)
    texts = []
    for _id, f in data.items():
        texts.append('{},{}'.format(_id, f))
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all co location summaries at {}'.format(output))
    del texts[:]
    data.clear()

    ### Extract raw co-occurrence data
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
    output = co_raw_filename.format(p, k, t, d)
    remove_file_if_exists(working_folder + output)
    write_to_file_buffered(working_folder + output, texts)
    debug('Finished writing all raw co location data at {}'.format(output))
    del texts[:]
    del texts

def extraction(p, k, t, d, working_folder):
    stat_f = {}     # frequency
    stat_d = {}     # diversity
    stat_ld = {}    # co-occurrence between user in each venue (distinct)
    stat_lv = {}    # co-occurrence between user in each venue
    stat_l = {}     # co-occurrence between user in each venue (dictionary)
    stat_t = {}     # time diff on co-occurrence
    stat_td = {}    # duration
    stat_ts = {}    # stability
    u_xy = {}       # u_xy = Average meeting time = delta_xy / |Theta_xy|
    week_num = {}   # Save a list of week number of co-occurrence between users
    ### Extract data from file
    fname = co_raw_filename.format(p, k, t, d)
    debug(fname)
    weeks = {}
    with open(working_folder + fname, 'r') as fr:
        for line in fr:
            split = line.strip().split(',')
            # user1,user2,vid,t_diff,frequency,time1,time2,t_avg
            if line.strip() == 'user1,user2,vid,t_diff,frequency,time1,time2,t_avg':
                continue
            u1 = split[0]
            u2 = split[1]
            vid = split[2]
            t_diff = int(split[3])
            t_avg = float(split[len(split)-1])
            friend = Friend(u1, u2)
            ### Extract frequency
            found = stat_f.get(friend)
            if found is None:
                found = 0
            stat_f[friend] = found + 1
            ### Extract diversity
            found = stat_l.get(friend)
            if found is None:
                found = {}
            found_vid = found.get(vid)
            if found_vid is None:
                found_vid = 0
            found_vid += 1
            found[vid] = found_vid
            stat_l[friend] = found
            ### Extract duration
            found = stat_t.get(friend)
            if found is None:
                found = []
            found.append(t_diff)
            stat_t[friend] = found
            ### Extract co-occurrence on a venue
            found = stat_lv.get(friend)
            if found is None:
                found = []
            found.append(vid)
            stat_lv[friend] = found
            if vid not in stat_lv:
                stat_ld[friend] = found
            found = week_num.get(friend)
            if found is None:
                found = []
            tanggal = date.fromtimestamp(t_avg)
            tahun = tanggal.year - 2009
            week = tanggal.isocalendar()[1] + (tahun * 53)
            found.append(week)
            week_num[friend] = found
    #         count = weeks.get(tanggal)
    #         if count is None:
    #             count = 0
    #         count += 1
    #         weeks[tanggal] = count
    # for w in sorted(weeks, key=weeks.get, reverse=True):
    #     debug('{}\t{}'.format(w, weeks[w]), clean=True)
    ### Extract diversity
    for friend, dictionary in stat_l.items():
        found = stat_d.get(friend)
        if found is None:
            found = []            
        for vid, x in dictionary.items():
            found.append(x)
        stat_d[friend] = found
    ### Extract duration
    duration = 0
    max_duration = 0
    for friend, data in stat_t.items():
        duration = max(data) - min(data)
        if duration > max_duration:
            max_duration = duration
        stat_td[friend] = duration
    for friend, data in stat_td.items():
        stat_td[friend] = float(stat_td[friend]) / float(max_duration)
        Theta_xy = stat_f.get(friend)
        if Theta_xy is None:
            debug('Error: Co-occurrence not found between {}'.format(friend))
            continue
        u_xy[friend] = float(stat_td.get(friend)) / float(Theta_xy)
    ### Extract stability
    sigma_z = {}    # Sum of t_xy^z-u_xy
    rho = {}        # standard deviation of co-occurrence time difference with average
    for friend, data in stat_t.items():
        for t in data:
            found = sigma_z.get(friend)
            if found is None:
                found = 0
            mean = u_xy.get(friend)
            if mean is None:
                debug('Error: mean u_xy not found between {}'.format(friend))
                continue
            if mean == 0:
                # debug('Error: mean u_xy is 0 between {}'.format(friend))
                continue
            found += pow( (float(t) / float(max_duration)) / mean, 2)
            sigma_z[friend] = found
        if sigma_z.get(friend) is None:
            # debug('Sigma z between {} is not found'.format(friend))
            continue
        if stat_ld.get(friend) is None:
            # debug('Distinct location count between {} is not found'.format(friend))
            continue
        rho[friend] = sqrt(sigma_z[friend] / len(stat_ld[friend]))
        if u_xy.get(friend) is None:
            continue
        stat_ts[friend] = exp(-1*(u_xy[friend]+rho[friend]))
        # debug(stat_ts[friend], clean=True)

    # for friend, weeks in week_num.items():
    #     debug('{}:{}'.format(friend, weeks))

    ### normalization of each weight
    # max_f = max(stat_f.values())
    # for friend, frequency in stat_f.items():
    #     stat_f[friend] = float(frequency) / max_f

    ### Debug
    ### Frequency
    # for friend, frequency in stat_f.items():
    #     debug('{},{}'.format(friend, frequency), clean=True)
    ### Entropy (Diversity)
    # for friend, data in stat_d.items():
    #     debug('{},{}'.format(friend, entropy(data)), clean=True)
    ### Duration
    # for friend, duration in stat_td.items():
    #     debug('{}\t{}'.format(friend, duration))
    ### Stability
    # for friend, weight in stat_ts.items():
    #     debug('{},{}'.format(friend, weight), clean=True)

    debug('Finished extracting co-occurrence features')

    return stat_f, stat_d, stat_td, stat_ts

def evaluation(friends, stat_f, stat_d, stat_td, stat_ts, p, k, t, d):
    summaries = {}

    ### Frequency
    max_val = max(stat_f.values())
    for friend, data in stat_f.items():
        stat_f[friend] = float(data) / max_val
    for friend, data in stat_f.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.frequency = data
        summaries[friend] = found

    ### Diversity
    max_val = 0
    for friend, data in stat_d.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.diversity = entropy(data)
        if found.diversity > max_val:
            max_val = found.diversity
        summaries[friend] = found
    for friend, evaluation in summaries.items():
        evaluation.diversity = float(evaluation.diversity) / max_val

    ### Duration
    max_val = max(stat_td.values())
    for friend, data in stat_td.items():
        stat_td[friend] = float(data) / max_val
    for friend, data in stat_td.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.duration = data
        summaries[friend] = found

    ### Stability
    max_val = max(stat_ts.values())
    for friend, data in stat_ts.items():
        stat_ts[friend] = float(data) / max_val
    for friend, data in stat_ts.items():
        found = summaries.get(friend)
        if found is None:
            found = Evaluation(friend.u1, friend.u2)
        found.stability = data
        summaries[friend] = found

    ### Friendship
    for uid, arr in friends.items():
        for fid in arr:
            friend = Friend(uid, fid)
            found = summaries.get(friend)
            if found is None:
                continue
            found.link = 1
            summaries[friend] = found
            # debug(found, clean=True)
    # for friend, evaluation in summaries.items():
    #     debug(evaluation, clean=True)
    write_evaluation(summaries, p, k, t, d)

# Main function
if __name__ == '__main__':
    ### For parallelization
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
    ts.append(1 * HOUR)
    # ts.append(int(1.5 * HOUR))
    # ts.append(2 * HOUR)
    # ts.append(1 * DAY)
    # ts.append(2 * DAY)
    # ts.append(3 * DAY)
    # ts.append(1 * WEEK)
    # ts.append(2 * WEEK)
    # ts.append(1 * MONTH)
    # ts.append(2 * MONTH)
    ### distance threshold to be included
    ds.append(0)
    # ds.append(100)
    # ds.append(250)
    # ds.append(500)
    # ds.append(750)
    # ds.append(1000)
    debug("--- Co-occurrence generation started ---")
    for p in ps:
        for k in ks:
            for t in ts:
                for d in ds:
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
                    # n_core = 4
                    # n_core = len(ss)
                    debug('Number of core: {}'.format(n_core))
                    Parallel(n_jobs=n_core)(delayed(mapping)(users, p, k, t, d, working_folder, ss[i], ff[i]) for i in range(len(ss)))
                    reducing(p, k, t, d, working_folder)
                    ### extracting features
                    # stat_f, stat_d, stat_td, stat_ts = extraction(p, k, t, d, working_folder)
                    # evaluation(friends, stat_f, stat_d, stat_td, stat_ts, p, k, t, d)
                    ### testing extracted csv
                    # testing(p, k, t, d, working_folder)
    debug("--- Co-occurrence generation finished ---")