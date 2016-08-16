import re
import os
import time
from general_utilities import *
from datetime import date
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

ACTIVE_PROJECT = 0 # 0 Gowalla dataset, 1 Brightkite dataset
IS_DEBUG = True
topk = 100
HOURDAY = 24
HOURWEEK = 24 * 7

class Venue:
    def __init__(self, _id, _count, _lat, _lon):
        self.id = _id               # Venue id
        self.count = _count
        self.lat = _lat
        self.lon = _lon

class Checkin:
    def __init__(self, _uid, _vid, _lat, _lon, _time):
        self.uid = _uid
        self.vid = _vid
        self.lat = _lat
        self.lon = _lon
        self.time = _time

class User:
    def __init__(self, _id):
        self.id = _id               # User id
        self.checkins = []

    def add_checkin(self, _vid, _lat, _lon, _time):
        self.checkins.append(Checkin(self.id, _vid, _lat, _lon, _time))

CHECKIN_FILE = 'checkin.csv'
FRIEND_FILE = 'friend.csv'
USER_FILE = 'user.csv'
VENUE_FILE = 'venue_location.csv'

dataset = ["gowalla", "brightkite"]
base_folder = "{0}/base/".format(dataset[ACTIVE_PROJECT])
working_folder = "{0}/working/".format(dataset[ACTIVE_PROJECT])

# Utility functions
def debug(message, callerid=None):
    if IS_DEBUG == False:
        return
    if callerid is None:
        print('[DEBUG] {0}'.format(message))
    else :
        print('[DEBUG] <Caller: {1}> {0}'.format(message, callerid))

# Initializiation functions
def init_checkins(file=None):
    if file is None:
        file = base_folder + CHECKIN_FILE
    users = {}
    counter = 0
    error = 0
    query_time = time.time()
    count_weekend = 0
    with open(file, 'r') as fcheck:
        for line in fcheck:
            split = line.strip().split(',')
            uid = int(split[0])
            timestamp = int(split[1])
            lat = float(split[2])
            lon = float(split[3])
            vid = int(split[4])
            if lat == 0.0 or lon == 0.0 or lat > 180 or lon > 180 or lat <-180 or lat <-180:
                error += 1
                continue
            user = users.get(uid)
            if user is None:
                user = User(uid)
                users[uid] = user
            user.add_checkin(vid, lat, lon, timestamp)
            counter += 1
            # Counting weekend checkins
            x = date.fromtimestamp(timestamp)
            if x.weekday() >= 5:
                count_weekend += 1
                # write_to_file(working_folder + 'checkin_weekend.csv', line.strip())
    process_time = int(time.time() - query_time)
    print('Processing {0:,} checkins in {1} seconds'.format(counter, process_time))
    print('There are {} errors in checkin file'.format(error))
    debug('Count weekend: {:,}'.format(count_weekend))
    # if IS_DEBUG:
    #     show_object_size(users, 'users')
    return users

def init_friendships(file=None):
    if file is None:
        file = base_folder + FRIEND_FILE
    friends = {}
    counter = 0
    query_time = time.time()
    with open(file, 'r') as fcheck:
        for line in fcheck:
            split = line.strip().split(',')
            if len(split) != 2:
                continue
            uid = int(split[0])
            fid = int(split[1])
            friend = friends.get(uid)
            if friend is None:
                friend = []
                friends[uid] = friend
            friend.append(fid)
            counter += 1
    process_time = int(time.time() - query_time)
    print('Processing {0:,} friendships in {1} seconds'.format(counter, process_time))
    return friends

def init():
    make_sure_path_exists(working_folder)
    checkins_file = working_folder + 'checkin{}.csv'.format(topk)
    checkins_file = None
    users = init_checkins(checkins_file)
    friends = init_friendships()
    return users, friends

def write_user_checkins_recap(users):
    with open(working_folder + 'user_checkins.csv', 'a') as fw:
        for uid, user in users.items():
            fw.write('{},{}\n'.format(uid, len(user.checkins)))

def select_top_k_users_checkins(users, topk):
    topk_users = {}
    uids = []
    with open(working_folder + 'user_checkins.csv', 'r') as fr:
        count = 0
        for line in fr:
            split = line.split(',')
            uid = int(split[0])
            count += 1
            if count >= topk:
                break
    for uid, user in users.items():
        if uid in topk_users:
            for checkin in user.checkins:
                with open(working_folder + 'checkin{}.csv'.format(topk), 'a') as fw:
                    fw.write('{},{},{},{},{}\n'.format(uid, checkin.time, checkin.lat, checkin.lon, checkin.vid))

def select_top_k_friendship(uids, friends, topk):
    with open(working_folder + 'friend{}.csv'.format(topk), 'w') as fw:
        for uid in uids:
            for friend in friends[uid]:
                if friend in uids:
                    fw.write('{},{}\n'.format(uid, friend))

def sort_user_checkins(users):
    uids = []
    for uid, user in users.items():
        user.checkins = sorted(user.checkins, key=lambda checkin: checkin.time, reverse=False)   # sort by checkin time
        uids.append(uid)
    return uids

def normalize_array(data):
    maxvalue = 0
    for i in range(0, len(data)):
        if maxvalue < data[i]:
            maxvalue = data[i]
    for i in range(0, len(data)):
        data[i] = float(data[i]) / maxvalue
    return data

def plot_hourly(uids, users_time_slots):
    # Show the 1st and 2nd top users' hourly checkins' distribution (based on their checkin)
    debug(users_time_slots[uids[0]])
    debug(users_time_slots[uids[1]])
    t = np.arange(1, HOURDAY+1, 1)
    plt.plot(t, users_time_slots[uids[0]], 'b-', t, users_time_slots[uids[1]], 'g-')
    plt.show()

def sim_difference(arr1, arr2):
    score = 0
    for i in range(0, len(arr1)):
        score += similarity(arr1[i], arr2[i])
    score = score / len(arr1)
    return score

def calculate_temporal_similarity(uid1, uid2, users_time_slots):
    u = users_time_slots[uid1]
    v = users_time_slots[uid2]
    # score = distance.euclidean(u, v)
    score = distance.cosine(u, v)
    # score = distance.correlation(u, v)
    # score = distance.cityblock(u, v)
    # score = distance.canberra(u, v)
    # score = distance.chebyshev(u, v)
    # score = distance.braycurtis(u, v)
    # score = distance.minkowski(u, v, 1)
    return score

# Main function
if __name__ == '__main__':
    print("--- Program  started ---")
    ### Initialize dataset
    users, friends = init()
    # debug('Number of users : {:,}'.format(len(users)), 'MAIN')
    
    ### Selecting topk users, for testing purpose
    # select_top_k_users_checkins(users, topk)
    
    ### Sorting users' checkins based on their timestamp, ascending ordering
    # uids = sort_user_checkins(users)
    ### Generate topk users' friendship
    # select_top_k_friendship(uids, friends, topk)

    ### Iterate over users' checkins
    # users_time_slots = {}
    # for uid, user in users.items():
    #     ### Prepare empty timeslots
    #     time_slots = []
    #     for i in range(0, HOURDAY):
    #         time_slots.append(0)
    #     ### Measure the time distributions of each user's checkins
    #     datetemp = None
    #     for c in user.checkins:
    #         x = datetime.fromtimestamp(c.time)
    #         if datetemp is not None:
    #             delta = x - datetemp
    #         datetemp = x
    #         time_slots[x.hour-1] += 1
    #     normalize_array(time_slots)
    #     users_time_slots[uid] = time_slots

    # plot_hourly(uids, users_time_slots)

    ### Capture the score of true friend and not
    # friend_true = []
    # friend_false = []
    # friend_threshold = {}
    # threshold = 0.299
    # for i in range(0, len(uids)):
    #     for j in range(i+1, len(uids)):
    #         score = calculate_temporal_similarity(uids[i], uids[j], users_time_slots)
    #         # debug('Score for \t{}\t{}\t{}\t{}'.format(uids[i], uids[j], score, uids[j] in friends[uids[i]]))
    #         if uids[j] in friends[uids[i]]:
    #             friend_true.append(score)
    #         else:
    #             friend_false.append(score)
    #         if score <= threshold:
    #             ft = friend_threshold.get(uids[i])
    #             if ft is None:
    #                 ft = []
    #                 friend_threshold[uids[i]] = ft
    #             ft.append(uids[j])

    # df_true = pd.DataFrame(friend_true)
    # df_false = pd.DataFrame(friend_false)

    # debug(df_true.describe())
    # debug(df_false.describe())

    # correct = 0
    # counter = 0
    # for uid, friend in friend_threshold.items():
    #     for f in friend:
    #         counter += 1
    #         if f in friends[uid]:
    #             correct += 1
    # debug('Correct: {}'.format(correct))
    # debug('All    : {}'.format(counter))

    print("--- Program finished ---")