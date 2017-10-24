import json
import time

from datetime import date
from classes import *
from general_utilities import *

base_folder = '' 
working_folder = '' 
weekend_folder = ''

CHECKIN_THRESHOLD = 10

def init_variables():
    filename = 'variables.json'
    global dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER
    with open(filename) as data_file:
        data = json.load(data_file)
        dataset = data["dataset"]
        CHECKIN_FILE    = data["filenames"]["checkin"]
        FRIEND_FILE     = data["filenames"]["friend"]
        USER_FILE       = data["filenames"]["user"]
        VENUE_FILE      = data["filenames"]["venue"]
        USER_DIST       = data["filenames"]["user_dist"]
        VENUE_CLUSTER   = data["filenames"]["venue_cluster"]
    return dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER

def init_config(filename):
    config = {}
    with open(filename) as data_file:
        data = json.load(data_file)
    return data 

def init_folder(ACTIVE_PROJECT, WEEKEND=False):
    global base_folder, working_folder, weekend_folder
    global CHECKIN_WEEKEND, FRIEND_WEEKEND, USER_WEEKEND, VENUE_WEEKEND
    global USER_DIST_WEEKEND, VENUE_CLUSTER_WEEKEND
    dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
    ### Update folder based on Active project
    ACTIVE_PROJECT = int(ACTIVE_PROJECT)
    base_folder = "{0}/base/".format(dataset[ACTIVE_PROJECT])
    working_folder = "{0}/working/".format(dataset[ACTIVE_PROJECT])
    weekend_folder = "{0}/weekend/".format(dataset[ACTIVE_PROJECT])

    if WEEKEND == True:
        CHECKIN_WEEKEND = weekend_folder + CHECKIN_FILE
        FRIEND_WEEKEND = weekend_folder + FRIEND_FILE
        USER_WEEKEND = weekend_folder + USER_FILE
        VENUE_WEEKEND = weekend_folder + VENUE_FILE
        USER_DIST_WEEKEND = weekend_folder + USER_DIST
        VENUE_CLUSTER_WEEKEND = weekend_folder + VENUE_CLUSTER

    return dataset, base_folder, working_folder, weekend_folder

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
    debug('Processing {0:,} checkins in {1} seconds'.format(counter, process_time))
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
            friend = Friend(uid, fid)
            if uid in users and fid in users:
                friend = friends.get(uid)
                if friend is None:
                    friend = []
                    friends[uid] = friend
                friend.append(fid)
            counter += 1
    process_time = int(time.time() - query_time)
    debug('Processing {0:,} friendships in {1} seconds'.format(counter, process_time))
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
    debug('Processing {0:,} venues in {1} seconds'.format(counter, process_time))
    debug('There are {} errors in venue file'.format(error))
    return venues

def filter_users(users):
    output = {k: v for k, v in users.items() if v.get_ncheckin() >= CHECKIN_THRESHOLD}
    if IS_DEBUG:
        print('Filtering users')
        print(len(output))
        print(len(users))
    users.clear()
    del users
    return output

def init(ACTIVE_PROJECT, topk):
    debug("Starting initialization")
    WEEKEND = False
    if topk == 0:
        WEEKEND = True
    dataset, base_folder, working_folder, weekend_folder = init_folder(ACTIVE_PROJECT, WEEKEND)
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

    users = filter_users(users)
    return users, friends, venues

def sort_user_checkins(users):
    uids = []
    query_time = time.time()
    for uid, user in users.items():
        user.checkins.sort(key=lambda checkin: checkin.time, reverse=False)   # sort by checkin time
        uids.append(uid)
    process_time = int(time.time() - query_time)
    debug('Sorting {0:,} users in {1} seconds'.format(len(users), process_time))
    return uids

def sort_colocation(colocations):
    query_time = time.time()
    for friend, co in colocations.items():
        co = sorted(co, key=lambda co: co.t_avg, reverse=False)   # sort by co-occurrence time
    process_time = int(time.time() - query_time)
    debug('Sorting {0:,} colocations in {1} seconds'.format(len(colocations), process_time))