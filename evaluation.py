from general_utilities import *
from base import *
from classes import *

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import precision_recall_curve
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from scipy import interp

import numpy as np
import time

def generate_report(X, y, Xs, notes, p, k, t, d):
    texts = []
    for idx in range(len(Xs)):
        Xi = Xs[idx]
        debug('Evaluating {}'.format(notes[idx]))
        lists, names = sampling(X, y)
        for i in range(len(lists)):
            (Xii, yii) = lists[i]
            name = names[i]
            debug('Evaluating {}'.format(name))
            mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue = auc_score(Xii, yii)
            text = '{},{},{},{},{:.9f},{:.9f},{:.9f},{:.9f},{},{},{},{}'.format(
                p, k, t, d, 
                mean_auc, mean_precision, mean_recall, mean_f1, 
                total_ytrue, len(y),
                notes[idx],
                name
            )
            texts.append(text)
    return texts

def sampling(X, y):
    debug('Started sampling')
    lists = []
    names = []
    lists.append((X, y))
    names.append('original')

    ### ovesampling
    # query_time = time.time()
    # pp = SMOTE(kind='regular')
    # X_pp, y_pp = pp.fit_sample(X, y)
    # lists.append((X_pp, y_pp))
    # names.append('over-SMOTE')
    # process_time = int(time.time() - query_time)
    # debug('Finished sampling SMOTE in {} seconds'.format(process_time))

    ### undersampling
    # query_time = time.time()
    # pp = EditedNearestNeighbours()
    # X_pp, y_pp = pp.fit_sample(X, y)
    # lists.append((X_pp, y_pp))
    # names.append('under-ENN')
    # process_time = int(time.time() - query_time)
    # debug('Finished sampling ENN in {} seconds'.format(process_time))
    
    ### oversampling + undersampling
    # query_time = time.time()
    # pp = SMOTEENN()
    # X_pp, y_pp = pp.fit_sample(X, y)
    # lists.append((X_pp, y_pp))
    # names.append('over+under-SMOTE-ENN')
    # process_time = int(time.time() - query_time)
    # debug('Finished sampling SMOTE-ENN in {} seconds'.format(process_time))
    
    return lists, names

def auc_score(X, y):
    cv = StratifiedKFold(n_splits=5)
    # clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=3, random_state=0, n_jobs=1)
    clf = RandomForestClassifier(n_jobs=1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0

    total_ytrue = sum(y)

    # debug(len(X))
    # debug(len(y))

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
    debug(len(y))

    return mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue