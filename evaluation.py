from general_utilities import *
from base import *
from classes import *

import time
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsemble

from numpy import genfromtxt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

evaluation_filename = 'evaluation_p{}_k{}_t{}_d{}.csv'

def cv_score(X, y):
    ### using 2 cores (n_jobs = 2)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=3, random_state=0, n_jobs=2)
    scores = cross_val_score(clf, X, y, cv=5)
    # print(scores)
    score = scores.mean()
    # print(score)
    debug('Finished evaluating cross validation scores')
    return score

def sampling(X, y):
    lists = []
    lists.append((X, y))
    ### ovesampling
    sm = SMOTE(kind='regular')
    X_smote, y_smote = sm.fit_sample(X, y)
    lists.append((X_smote, y_smote))
    ### undersampling
    sm = SMOTEENN()
    X_combine, y_combine = sm.fit_sample(X, y)
    lists.append((X_combine, y_combine))
    debug('Finished sampling')
    return lists

def testing(p, k, t, d, working_folder):
    filename = working_folder + evaluation_filename.format(p, k, t, d)
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(filename, delimiter=',')[1:]
    # print(dataset.shape)
    ncol = dataset.shape[1]
    X = dataset[:,0:ncol-2]
    y = dataset[:,ncol-1]
    lists = sampling(X, y)
    scores = []
    for Xi, yi in lists:
        score = cv_score(Xi, yi)
        scores.append(score)
    debug(scores)

# Main function
if __name__ == '__main__':
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
    debug("--- Evaluation started ---")
    for p in ps:
        for k in ks:
            for t in ts:
                for d in ds:
                    debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d))
                    ### Initialize variables
                    dataset, base_folder, working_folder, weekend_folder = init_folder(p)
                    # dataset, CHECKIN_FILE, FRIEND_FILE, USER_FILE, VENUE_FILE, USER_DIST, VENUE_CLUSTER = init_variables()
                    # ### Initialize dataset
                    # users, friends, venues = init(p, k)
                    # ### Sorting users' checkins based on their timestamp, ascending ordering
                    # uids = sort_user_checkins(users)
                    testing(p, k, t, d, working_folder)
    debug("--- Evaluation finished ---")