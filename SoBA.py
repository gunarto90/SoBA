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

from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan  # https://github.com/lmcinnes/hdbscan
import seaborn as sns


ACTIVE_PROJECT = 0 # 0 Gowalla dataset, 1 Brightkite dataset
IS_DEBUG = True
topk = 100
HOURDAY = 24
HOURWEEK = 24 * 7

class Venue:
    def __init__(self, _id, _lat, _lon):
        self.id = _id               # Venue id
        self.lat = _lat
        self.lon = _lon
        self.count = 0
        self.cluster = -1

    def set_count(self, count):
        self.count = count

    def increase_count(self):
        self.count += 1

    def set_cluster(self, cluster_id):
        self.cluster = cluster_id

    def __str__(self):
        return '{},{},{}'.format(self.id, self.lat, self.lon)

class Checkin:
    def __init__(self, _uid, _vid, _lat, _lon, _time):
        self.uid = _uid
        self.vid = _vid
        self.lat = _lat
        self.lon = _lon
        self.time = _time

    def __str__(self):
        return '{},{},{},{},{}'.format(self.uid, self.time, self.lat, self.lon, self.vid)

class User:
    def __init__(self, _id):
        self.id = _id               # User id
        self.checkins = []

    def add_checkin(self, _vid, _lat, _lon, _time):
        self.checkins.append(Checkin(self.id, _vid, _lat, _lon, _time))

    def __str__(self):
        return '{},{}'.format(self.id, len(self.checkins))

dataset = ["gowalla", "brightkite"]
base_folder = "{0}/base/".format(dataset[ACTIVE_PROJECT])
working_folder = "{0}/working/".format(dataset[ACTIVE_PROJECT])
weekend_folder = "{0}/weekend/".format(dataset[ACTIVE_PROJECT])

CHECKIN_FILE = 'checkin.csv'
FRIEND_FILE = 'friend.csv'
USER_FILE = 'user.csv'
VENUE_FILE = 'venue_location.csv'

CHECKIN_WEEKEND = working_folder + 'checkins_weekend.csv'

# Utility functions
def debug(message, callerid=None):
    if IS_DEBUG == False:
        return
    if callerid is None:
        print('[DEBUG] {0}'.format(message))
    else :
        print('[DEBUG] <Caller: {1}> {0}'.format(message, callerid))

# Initializiation functions
def init_checkins(venues, file=None):
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
                venue = venues.get(vid)
                if venue is not None:
                    venue.increase_count()
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

def init_friendships(users, file=None):
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
            if uid in users and fid in users:
                friend = friends.get(uid)
                if friend is None:
                    friend = []
                    friends[uid] = friend
                friend.append(fid)
            counter += 1
    process_time = int(time.time() - query_time)
    print('Processing {0:,} friendships in {1} seconds'.format(counter, process_time))
    return friends

def init_venues(file=None):
    if file is None:
        file = base_folder + VENUE_FILE
    venues = {}
    counter = 0
    error = 0
    query_time = time.time()
    with open(file, 'r') as fcheck:
        for line in fcheck:
            split = line.strip().split(',')
            vid = int(split[0])
            lat = float(split[1])
            lon = float(split[2])
            if lat == 0.0 or lon == 0.0 or lat > 180 or lon > 180 or lat <-180 or lat <-180:
                error += 1
                continue
            v = venues.get(vid)
            if v is None:
                v = Venue(vid, lat, lon)
                venues[vid] = v
            else:
                v.lat = (v.lat + lat) /2
                v.lon = (v.lon + lon) /2
            counter += 1
    process_time = int(time.time() - query_time)
    print('Processing {0:,} venues in {1} seconds'.format(counter, process_time))
    print('There are {} errors in venue file'.format(error))
    return venues

def init():
    make_sure_path_exists(working_folder)
    ### Top k users' checkins
    checkins_file = working_folder + 'checkin{}.csv'.format(topk)
    ### Weekend checkins
    checkins_file = CHECKIN_WEEKEND
    ### Extracted venue
    venue_file = working_folder + VENUE_FILE

    ### Original files
    # checkins_file = None
    # venue_file = None

    ### Run initialization
    venues = init_venues(venue_file)
    users = init_checkins(venues, checkins_file)
    friends = init_friendships(users)
    return users, friends, venues

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

def clustering_venues(X):
    debug("Running clustering")
    query_time = time.time()
    EPS = 0.3
    MIN_SAMPLES = 10
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
    process_time = int(time.time() - query_time)
    debug('Finished clustering {0:,} venues in {1} seconds'.format(len(X), process_time))

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # debug(labels)

    count_outlier = 0
    for x in labels:
        if x == -1:
            count_outlier += 1
    debug('#Labels: {}'.format(len(labels)))
    debug('#Outlier : {}'.format(count_outlier))

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    debug('Number of clusters: {}'.format(n_clusters_))

    ### Plot clusters
    # import matplotlib.pyplot as plt
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = 'k'

    #     class_member_mask = (labels == k)

    #     ### Clustered results
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)

    #     ### Outlier ?
    #     # xy = X[class_member_mask & ~core_samples_mask]
    #     # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #     #          markeredgecolor='k', markersize=6)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

def hdcluster(X):
    debug("Running clustering")
    query_time = time.time()
    MIN_CLUSTER_SIZE = 10
    MIN_SAMPLES = 3
    db = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit(X)

    labels = db.labels_
    # labels = db.fit_predict(X)
    process_time = int(time.time() - query_time)
    debug('Finished clustering {0:,} venues in {1} seconds'.format(len(X), process_time))

    count_outlier = 0
    for x in labels:
        if x == -1:
            count_outlier += 1
    debug('#Labels: {}'.format(len(labels)))
    debug('#Outlier : {}'.format(count_outlier))

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    debug('Number of clusters: {}'.format(n_clusters_))

    return labels

# Main function
if __name__ == '__main__':
    print("--- Program  started ---")
    ### Initialize dataset
    users, friends, venues = init()
    # debug('Number of users : {:,}'.format(len(users)), 'MAIN')
    
    ### Selecting topk users, for testing purpose
    # select_top_k_users_checkins(users, topk)

    ### Process venue, user, and friendship in weekend data
    list_venue = []
    count = 0
    texts = []
    for vid, venue in venues.items():
        if venue.count > 0:
            count += 1
            # texts.append(str(venue))
    # write_to_file_buffered(working_folder + 'venue_weekend.csv', texts)
    debug(count)
    debug(len(venues))

    count = 0
    texts = []
    for uid, user in users.items():
        if len(user.checkins) > 0:
            count += 1
    #         texts.append(str(user))
    # write_to_file_buffered(working_folder + 'user_weekend.csv', texts)
    debug(count)
    debug(len(users))

    count = 0
    texts = []
    for uid, f_list in friends.items():
        for fid in f_list:
            count += 1
    #         texts.append('{},{}'.format(uid, fid))
    # write_to_file_buffered(working_folder + 'friend_weekend.csv', texts)
    debug(count)

    ### Normalizing venues
    # list_venue = []
    #     list_venue.append('{},{},{}'.format(vid, venue.lat, venue.lon))
    # debug(len(list_venue))
    # write_to_file_buffered(working_folder + VENUE_FILE, list_venue)

    ### Clustering venues
    # list_venue = []
    # for vid, venue in venues.items():
    #     temp = []
    #     # temp.append(vid)
    #     temp.append(venue.lat)
    #     temp.append(venue.lon)
    #     list_venue.append(temp)
    #     if len(list_venue) >= 100000:
    #         break
    # X = np.array(list_venue)
    # # clustering_venues(X)
    # cluster_labels = hdcluster(X)
    
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