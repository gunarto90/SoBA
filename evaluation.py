from general_utilities import *
from base import *
from classes import *

import time
import numpy as np

from scipy import interp

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsemble

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
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
    names = []
    lists.append((X, y))
    names.append('original')
    ### ovesampling
    sm = SMOTE(kind='regular')
    X_smote, y_smote = sm.fit_sample(X, y)
    lists.append((X_smote, y_smote))
    names.append('over-SMOTE')
    ### undersampling
    sm = SMOTEENN()
    X_combine, y_combine = sm.fit_sample(X, y)
    lists.append((X_combine, y_combine))
    names.append('over+under-SMOTE-ENN')
    debug('Finished sampling')
    return lists, names

def testing(p, k, t, d, working_folder):
    filename = working_folder + evaluation_filename.format(p, k, t, d)
    if is_file_exists(filename) is True:
        #create the training & test sets, skipping the header row with [1:]
        dataset = np.genfromtxt(filename, delimiter=',')[1:]
        # print(dataset.shape)
        ncol = dataset.shape[1]
        X = dataset[:,0:ncol-2]
        y = dataset[:,ncol-1]
        mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue = auc_score(X, y)
        # lists, names = sampling(X, y)
        # scores = {}
        # for i in range(len(lists)):
        #     (Xi, yi) = lists[i]
        #     name = names[i]
        #     score = cv_score(Xi, yi)
        #     scores[name] = score
        # debug(scores)

        return '{},{},{},{},{:.9f},{:.9f},{:.9f},{:.9f},{},{}'.format(p, k, t, d, mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue,len(y))
    return None

def auc_score(X, y):
    cv = StratifiedKFold(n_splits=10)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=3, random_state=0, n_jobs=2)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0

    total_ytrue = sum(y)

    i = 0
    for (train, test) in cv.split(X, y):
        fit = clf.fit(X[train], y[train])
        probas_ = fit.predict_proba(X[test])
        inference = fit.predict(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        # precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        average = 'weighted'
        precision = precision_score(y[test], inference, average=average)
        recall = recall_score(y[test], inference, average=average)
        f1 = f1_score(y[test], inference, average=average)

        mean_precision += precision
        mean_recall += recall
        mean_f1 += f1
        # debug(roc_auc)
        i += 1
    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    mean_precision /= cv.get_n_splits(X, y)
    mean_recall /= cv.get_n_splits(X, y)
    mean_f1 /= cv.get_n_splits(X, y)
    
    debug(mean_auc)
    debug(mean_precision)
    debug(mean_recall)
    debug(mean_f1)
    debug(total_ytrue)

    return mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue

# Main function
if __name__ == '__main__':
    ### Global parameter for the experiments
    ps = []     ### Active project: 0 Gowalla, 1 Brightkite
    ks = []     ### Mode for top k users: 0 Weekend, -1 All users
    ts = []     ### Time threshold
    ds = []     ### Distance threshold
    ### project to be included
    ps.append(0)
    ps.append(1)
    ### mode to be included
    ks.append(0)
    ks.append(-1)
    ### time threshold to be included
    HOUR  = 3600
    DAY   = 24 * HOUR
    WEEK  = 7 * DAY
    MONTH = 30 * DAY
    ts.append(int(0.5 * HOUR))
    ts.append(1 * HOUR)
    ts.append(int(1.5 * HOUR))
    ts.append(2 * HOUR)
    # ts.append(1 * DAY)
    # ts.append(2 * DAY)
    # ts.append(3 * DAY)
    # ts.append(1 * WEEK)
    # ts.append(2 * WEEK)
    # ts.append(1 * MONTH)
    # ts.append(2 * MONTH)
    ### distance threshold to be included
    ds.append(0)
    ds.append(250)
    ds.append(500)
    ds.append(750)
    # ds.append(1000)
    debug("--- Evaluation started ---")
    header = 'p,k,t,d,auc,precision,recall,f1,link_found,all_data'
    for p in ps:
        dataset, base_folder, working_folder, weekend_folder = init_folder(p)
        result_filename = working_folder + 'result.csv'
        debug(result_filename)
        remove_file_if_exists(result_filename)
        write_to_file(result_filename, header)
        for k in ks:
            for t in ts:
                for d in ds:
                    debug('p:{}, k:{}, t:{}, d:{}'.format(p, k, t, d))
                    ### Initialize variables
                    text = testing(p, k, t, d, working_folder)
                    if text is not None:
                        write_to_file(result_filename, text)
    debug("--- Evaluation finished ---")