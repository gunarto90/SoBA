from general_utilities import *
from base import *
from classes import *

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

co_raw_filename = 'co_raw_p{}_k{}_t{}_d{}.csv'
co_location_filename = 'co_location_p{}_k{}_t{}_d{}.csv'
evaluation_filename = 'evaluation_p{}_k{}_t{}_d{}.csv'

friend_profile_filename = 'data_friend_checkin_profile_p{}_k{}.csv'

def friend_profile(friends, users, p, k):
    texts = []
    for uid, friend in friends.items():
        for fid in friend:
            n1 = len(users.get(uid).checkins)
            n2 = len(users.get(fid).checkins)
            lat1 = sum(c.lat for c in users.get(uid).checkins)/n1
            lon1 = sum(c.lon for c in users.get(uid).checkins)/n1
            lat2 = sum(c.lat for c in users.get(fid).checkins)/n2
            lon2 = sum(c.lon for c in users.get(fid).checkins)/n2
            time1 = sum(c.time for c in users.get(uid).checkins)/n1
            time2 = sum(c.time for c in users.get(fid).checkins)/n2
            text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}'.format(
                uid, fid, 
                n1, n2,
                lat1, lon1, lat2, lon2, time1, time2,
                haversine(lat1, lon1, lat2, lon2), abs(time1-time2)
                )
            texts.append(text)
    out_file = working_folder + friend_profile_filename.format(p,k)
    remove_file_if_exists(out_file)
    write_to_file_buffered(out_file, texts)

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def plot_friend_profile(filename):
    data = np.genfromtxt(filename, delimiter=',')
    # debug(data[:3])
    debug(data.shape)
    ncol = data.shape[1]
    X = data[:,2:ncol-1] # Remove user 1 and user 2
    n1 = X[:,0]
    n2 = X[:,1]
    dist = X[:,7]
    timediff = X[:,8]

    plot_histogram(n1)
    # plot_histogram(n2)
    # plot_histogram(dist)
    # plot_histogram(timediff)

def plot_histogram(X):
    matplotlib.rcParams.update({'font.size': 18})
    plt.hist(X, bins=50, normed=True)
    plt.axis([0, 1000, 0.00, 0.03])
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([0, 25, 50, 100, 200, 500, 1000])
    frame1.axes.get_yaxis().set_ticks([0.00, 0.015, 0.03])
    # plt.ylabel('Ratio of users')
    # plt.xlabel('#Checkins')
    # formatter = FuncFormatter(to_percent)
    # Set the formatter
    # plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()

# Main function
if __name__ == '__main__':
    """
    0 : Extract friend profile
    1 : Plot friend profile
    """
    MODE = 1
    ### Global parameter for the experiments
    ps = []     ### Active project: 0 Gowalla, 1 Brightkite
    ks = []     ### Mode for top k users: 0 Weekend, -1 All users
    ### project to be included
    ps.append(0)
    ps.append(1)
    ### mode to be included
    ks.append(0)
    ks.append(-1)
    debug("--- Test started ---")
    for p in ps:
        for k in ks:
            debug('p:{}, k:{}'.format(p, k))
            ### Initialize variables
            dataset, base_folder, working_folder, weekend_folder = init_folder(p)
            dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
            if MODE == 0:
                # ### Initialize dataset
                users, friends, venues = init(p, k)
                # ### Sorting users' checkins based on their timestamp, ascending ordering
                uids = sort_user_checkins(users)
                friend_profile(friends, users, p, k)
            elif MODE == 1:
                plot_friend_profile(working_folder + friend_profile_filename.format(p,k))
    debug("--- Test finished ---")