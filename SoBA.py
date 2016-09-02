import re
import os
import time
import sys
import getopt
from general_utilities import *
from datetime import date
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.spatial import distance

# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# import hdbscan  # https://github.com/lmcinnes/hdbscan
# import seaborn as sns

ACTIVE_PROJECT = 0  # 0 Gowalla dataset, 1 Brightkite dataset
topk = 50           # 0 weekend, -1 all
HOURDAY = 24
HOURWEEK = 24 * 7
i_start = 0
BACKUP = 100
CO_TIME = 3600
CO_DISTANCE = 500

dataset = ["gowalla", "brightkite"]
CHECKIN_FILE = 'checkin.csv'
FRIEND_FILE = 'friend.csv'
USER_FILE = 'user.csv'
VENUE_FILE = 'venue.csv'

class Venue:
    def __init__(self, _id, _lat, _lon):
        self.vid = _id               # Venue id
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
        return '{},{},{}'.format(self.vid, self.lat, self.lon)

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
        self.uid = _id              # User id
        self.checkins = []
        self.friends = []
        self.dist = []
        self.earliest = sys.maxsize # Earliest checkin
        self.latest = 0             # Latest checkin

    def add_checkin(self, _vid, _lat, _lon, _time):
        self.checkins.append(Checkin(self.uid, _vid, _lat, _lon, _time))
        if _time < self.earliest:
            self.earliest = _time
        if _time > self.latest:
            self.latest = _time

    def add_friends(self, friend_list):
        for fid in friend_list:
            self.friends.append(fid)

    def add_distribution(self, venues, n_clusters):
        self.dist = []
        for i in range(0, n_clusters):
            self.dist.append(0)
        for c in self.checkins:
            vid = c.vid
            venue = venues.get(vid)
            if venue is None or venue.cluster == -1:
                continue
            self.dist[venue.cluster] += 1

    def __str__(self):
        return '{},{}'.format(self.id, len(self.checkins))

def init_folder():
    global base_folder, working_folder, weekend_folder
    global CHECKIN_WEEKEND, FRIEND_WEEKEND, USER_WEEKEND, VENUE_WEEKEND
    global USER_DIST_WEEKEND, VENUE_CLUSTER_WEEKEND
    ### Update folder based on Active project
    base_folder = "{0}/base/".format(dataset[ACTIVE_PROJECT])
    working_folder = "{0}/working/".format(dataset[ACTIVE_PROJECT])
    weekend_folder = "{0}/weekend/".format(dataset[ACTIVE_PROJECT])
    
    CHECKIN_WEEKEND = weekend_folder + CHECKIN_FILE
    FRIEND_WEEKEND = weekend_folder + FRIEND_FILE
    USER_WEEKEND = weekend_folder + USER_FILE
    VENUE_WEEKEND = weekend_folder + VENUE_FILE
    USER_DIST_WEEKEND = weekend_folder + 'user_dist.csv'
    VENUE_CLUSTER_WEEKEND = weekend_folder + 'venue_cluster.csv'

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
    debug('There are {} errors in checkin file'.format(error))
    # debug('Count weekend: {:,}'.format(count_weekend))
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
    debug('There are {} errors in venue file'.format(error))
    return venues

def init():
    init_folder()
    make_sure_path_exists(working_folder)
    ### Top k users' checkins
    checkins_file = working_folder + 'checkin{}.csv'.format(topk)
    ### Weekend checkins
    if topk == 0:
        checkins_file = CHECKIN_WEEKEND
    ### Extracted venue
    venue_file = base_folder + VENUE_FILE
    ### Venue weekend
    if topk == 0:
        venue_file = VENUE_WEEKEND
    ### Friends
    friend_file = working_folder + 'friend{}.csv'.format(topk)
    ### Friendship weekend
    if topk == 0:
        friend_file = FRIEND_WEEKEND

    ### Original files
    if topk == -1:
        checkins_file = None
        venue_file = None
        friend_file = None

    ### Run initialization
    venues = init_venues(venue_file)
    users = init_checkins(venues, checkins_file)
    friends = init_friendships(users, friend_file)
    return users, friends, venues

def write_user_checkins_recap(users):
    with open(working_folder + 'user_checkins.csv', 'a') as fw:
        for uid, user in users.items():
            fw.write('{},{}\n'.format(uid, len(user.checkins)))

def select_top_k_users_checkins(users, topk):
    topk_users = {}
    with open(working_folder + 'user_checkins.csv', 'r') as fr:
        count = 0
        for line in fr:
            split = line.split(',')
            uid = int(split[0])
            topk_users[uid] = 1
            count += 1
            if count >= topk:
                break
    for uid, user in users.items():
        if uid in topk_users:
            for checkin in user.checkins:
                with open(working_folder + 'checkin{}.csv'.format(topk), 'a') as fw:
                    fw.write('{},{},{},{},{}\n'.format(uid, checkin.time, checkin.lat, checkin.lon, checkin.vid))

def select_top_k_friendship(uids, friends, topk):
    topk_users = {}
    with open(working_folder + 'user_checkins.csv', 'r') as fr:
        count = 0
        for line in fr:
            split = line.split(',')
            uid = int(split[0])
            topk_users[uid] = 1
            count += 1
            if count >= topk:
                break
    with open(working_folder + 'friend{}.csv'.format(topk), 'w') as fw:
        for uid in topk_users:
            u_friend = friends.get(uid)
            if u_friend is None:
                debug('Friends of uid {} are not found'.format(uid), 'select_top_k_friendship')
                continue
            for friend in u_friend:
                if friend in uids:
                    fw.write('{},{}\n'.format(uid, friend))

def select_weekend_data(users, friends, venues):
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
    texts_f = []
    count_f = 0
    for uid, user in users.items():
        if len(user.checkins) > 0:
            count += 1
    #         texts.append(str(user))
        if len(user.friends) > 0:
            for fid in f_list:
                count_f += 1
                # texts_f.append('{},{}'.format(uid, fid))
    # write_to_file_buffered(working_folder + 'user_weekend.csv', texts)
    # write_to_file_buffered(working_folder + 'friend_weekend.csv', texts_f)
    debug(count)
    debug(len(users))
    debug(count_f)

def sort_user_checkins(users):
    uids = []
    query_time = time.time()
    for uid, user in users.items():
        user.checkins = sorted(user.checkins, key=lambda checkin: checkin.time, reverse=False)   # sort by checkin time
        uids.append(uid)
    process_time = int(time.time() - query_time)
    print('Sorting {0:,} users in {1} seconds'.format(len(users), process_time))
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

def dbscan(X):
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

    # ### Plot clusters
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

    return labels, n_clusters_

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

    return labels, n_clusters_

def clustering(venues, users, outputToFile=False):
    ### Clustering venues
    list_venue = []
    for vid, venue in venues.items():
        temp = []
        temp.append(venue.lat)
        temp.append(venue.lon)
        list_venue.append(temp)
        # if len(list_venue) >= 100000:
        #     break
    X = np.array(list_venue)
    cluster_labels, n_clusters = dbscan(X)
    # cluster_labels = hdcluster(X)

    ### Note: Cluster labels are from -1 (outlier) to the highest cluster id

    ### Set cluster ID
    if outputToFile is True:
        texts = []
    count = 0
    for vid, venue in venues.items():
        venue.set_cluster(cluster_labels[count])
        if outputToFile is True:
            texts.append('{},{}'.format(vid, cluster_labels[count]))
        count += 1
    if outputToFile is True:
        write_to_file_buffered(working_folder + 'venue_weekend_cluster.csv', texts)

    ### Count how the distribution of user's checkins among cluster
    query_time = time.time()
    for uid, user in users.items():
        user.add_distribution(venues, n_clusters)
    process_time = int(time.time() - query_time)
    debug('Finished adding users checkins distribution {} seconds'.format(process_time))

    ### Writing the result of user's checkin distribution
    if outputToFile is True:
        texts = []
        for uid, user in users.items():
            texts.append('{},{}'.format(uid, ','.join(str(x) for x in user.dist)))
        write_to_file_buffered(working_folder + 'user_weekend_cluster_dist.csv', texts)

def user_cluster_similarity():
    ### Open user cluster distribution file
    users = {}
    with open(USER_DIST_WEEKEND) as fr:
        for line in fr:
            split = line.split(',')
            uid = int(split[0])
            dist = []
            for i in range(1, len(split)):
                dist.append(int(split[i]))
            users[uid] = dist

    ### Calculate similarities between users
    for uid1, dist1 in users.items():
        del dist1[0]
        for uid2, dist2 in users.items():
            if uid1 == uid2:
                continue
            del dist2[0]
            score = distance.cityblock(dist1, dist2)
            print(sum(dist1))
            print(sum(dist2))
            print('{},{},{}'.format(uid1, uid2, score))
            break
        break

def extract_temporal(users):
    ### Iterate over users' checkins
    users_time_slots = {}
    query_time = time.time()
    for uid, user in users.items():
        ### Prepare empty timeslots
        time_slots = []
        for i in range(0, HOURDAY):
            time_slots.append(0)
        ### Measure the time distributions of each user's checkins
        datetemp = None
        for c in user.checkins:
            x = datetime.fromtimestamp(c.time)
            if datetemp is not None:
                delta = x - datetemp
            datetemp = x
            time_slots[x.hour-1] += 1
        normalize_array(time_slots)
        users_time_slots[uid] = time_slots
    process_time = int(time.time() - query_time)
    print('Extracting temporal pattern of {0:,} users in {1} seconds'.format(len(users), process_time))
    # plot_hourly(uids, users_time_slots)
    return users_time_slots

def write_co_location(co_location):
    ### Write to file
    texts = []
    texts.append('uid1,uid2,vid,frequency')
    for ss, frequency in co_location.items():
        texts.append('{},{}'.format(ss, frequency))
    filename = working_folder + 'co_location_p{}_k{}_s{}_t{}_d{}.csv'.format(ACTIVE_PROJECT, topk, i_start, CO_TIME, CO_DISTANCE)
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

def co_occur(users):
    """
    Time and distance threshold
    Time (in seconds)
    Distance (in meters)
    """
    t_threshold = CO_TIME
    d_threshold = CO_DISTANCE

    query_time = time.time()
    co_location = {}
    all_user = []
    for uid1, user in users.items():
        all_user.append(user)
    count = 0
    for i in range(i_start, len(all_user)):
        user1 = all_user[i]
        # print('{} of {} users ({}%) [{}]'.format(i, len(all_user), float(i)*100/len(all_user), datetime.now()))
        if i % 10 == 0:
            print('{} of {} users ({}%) [{}]'.format(i, len(all_user), float(i)*100/len(all_user), datetime.now()))
        if BACKUP > 0:
            if i > 0 and i % BACKUP == 0:
                ### Save current progress
                with open(working_folder + 'last_i.txt', 'w') as fi:
                    fi.write(str(i))
                write_co_location(co_location)
        for j in range(i+1, len(all_user)):
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
                # print('[A]:{} ({}), [B]:{} ({})'.format(ic1, len(user1.checkins), ic2, len(user2.checkins)))
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

# Main function
if __name__ == '__main__':
    print("--- Program  started ---")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"p:k:s:",["project=","topk=","start=","backup=","distance=","time="])
    except getopt.GetoptError:
        err_msg = 'SoBA.py -p <0 gowalla/ 1 brightkite> -k <top k users> -s <start position> [optional] --backup=<every #users to backup> --distance=<co-location distance threshold> --time=<co-location time threshold>'
        debug(err_msg, 'opt error')
        sys.exit(2)
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
        elif opt == "--backup":
            BACKUP = int(arg)
        elif opt == "--distance":
            CO_DISTANCE = int(arg)
        elif opt == "--time":
            CO_TIME = int(arg)
    debug('Selected project: {}'.format(dataset[ACTIVE_PROJECT]))
    debug('Starting iteration: {}'.format(i_start))
    debug('Backup every {} users'.format(BACKUP))
    debug('Co-location time threshold: {}'.format(CO_TIME))
    debug('Co-location distance threshold: {}'.format(CO_DISTANCE))
    if topk > 0:
        debug('Top {} users are selected'.format(topk))
    elif topk == 0:
        debug('Evaluating weekend checkins')
    elif topk == -1:
        debug('Evaluating all checkins')
    
    ### Initialize dataset
    users, friends, venues = init()
    # debug('Number of users : {:,}'.format(len(users)), 'MAIN')

    ### Sorting users' checkins based on their timestamp, ascending ordering
    uids = sort_user_checkins(users)
    
    ### Selecting topk users, for testing purpose
    # select_top_k_users_checkins(users, topk)

    ### Process venue, user, and friendship in weekend data (Preprocessing)
    # select_weekend_data(users, friends, venues)

    ### Normalizing venues
    # list_venue = []
    #     list_venue.append('{},{},{}'.format(vid, venue.lat, venue.lon))
    # debug(len(list_venue))
    # write_to_file_buffered(working_folder + VENUE_FILE, list_venue)

    ### Perform clustering on venues
    # clustering(venues, users, outputToFile=False)
    
    ### Generate topk users' friendship
    # select_top_k_friendship(uids, friends, topk)

    ### Co-location
    co_occur(users)

    # user_cluster_similarity()

    # users_time_slots = extract_temporal(users)
    ### Capture the score of true friend and not
    texts = []

    print("--- Program finished ---")