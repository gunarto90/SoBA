#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import os
import time
### Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
### Imbalance learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
### Setup Directories for local library
PWD = os.getcwd()
sys.path.append(PWD)
### Local libraries
from common.functions import read_config, debug, fn_timer, \
    make_sure_path_exists, is_file_exists, remove_file_if_exists

def sci_evaluation(config, p, k, t, d):
    debug('Evaluating SCI for p{}, k{}, t{}, d{}'.format(p, k, t, d))
    dataset_names = config['dataset']
    compressed = config['kwargs']['sci_eval']['read_compressed']
    sci_root = config['directory']['sci']
    make_sure_path_exists('/'.join([sci_root, dataset_names[p]]))
    if compressed is True:
        sci_name = config['intermediate']['sci']['evaluation_compressed']
    else:
        sci_name = config['intermediate']['sci']['evaluation']
    evaluation_name = '/'.join([sci_root, dataset_names[p], sci_name.format(p, k, t, d)])
    if is_file_exists(evaluation_name) is True:
        dataset = pd.read_csv(evaluation_name)
        # Format: 'uid1', 'uid2', 'frequency', 'diversity', 'duration', 'stability', 'popularity', 'link'
        X = dataset[['frequency', 'diversity', 'duration', 'stability', 'popularity']].values
        y = dataset[['link']].values
        ### Selecting the feature set
        selected_feature_set = config['kwargs']['sci_eval']['features']
        ### PAKDD 2017 Submission
        if selected_feature_set == 'pakdd_2017_all':
            notes = ["SCI", "frequency", "diversity", "duration", "stability", 
                "F+D", "F+TD", "F+TS", "D+TD", "D+TS", "TD+TS", 
                "F+D+TD", "F+D+TS", "F+TD+TS", "D+TD+TS"]
            assign = [[0,1,2,3], [0], [1], [2], [3], 
                [0,1], [0,2],[ 0,3], [1,2], [1,3], [2,3], 
                [0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        ### PAKDD 2017 Submission (All)
        elif selected_feature_set == 'pakdd_2017_summary':
            notes = ['SCI']
            assign = [ [0,1,2,3] ]
        ## New Feature added (Popularity)
        elif selected_feature_set == 'all_features':
            notes = ['SCI+', 'Frequency', 'Diversity', 'Duration', 'Stability', 'Popularity', 
                'F+D', 'F+TD', 'F+TS', 'F+P', 'D+TD', 'D+TS', 'D+P', 'TD+TS', 'TD+P', 'TS+P', 
                'F+D+TD', 'F+D+TS', 'F+D+P', 'F+TD+TS', 'F+TD+P', 
                'F+TS+P', 'D+TD+TS', 'D+TD+P', 'D+TS+P', 'TD+TS+P', 
                'F+D+TD+TS', 'F+D+TD+P', 'F+D+TS+P', 'F+TD+TS+P', 'D+TD+TS+P', 'SCI']
            assign = [ [0,1,2,3,4], [0], [1], [2], [3], [4], 
                [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], 
                [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], 
                [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], 
                [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0,1,2,3] ]
        ### Only All features
        elif selected_feature_set == 'summary':
            notes = ['SCI+']
            assign = [ [0,1,2,3,4] ]
        ### SCI and SCI+
        else:   ### 'summary_old_new'
            notes = ['SCI+', 'SCI']
            assign = [ [0,1,2,3,4], [0,1,2,3] ]
        ### Generate the report
        debug(notes, assign)
        texts = generate_report(config, X, y, assign, notes, p, k, t, d)
        del X, y
        report_directory = config['directory']['report']
        result_filename = '/'.join([report_directory, 'SCI_result_p{}_k{}.csv'.format(p,k)])
        for text in texts:
            if text is not None:
                with open(result_filename, 'ab') as fw:
                    fw.write(text+'\n')
    else:
        debug('File not found', evaluation_name)

def generate_report(config, X, y, assign, notes, p, k, t, d):
    texts = []
    names = config['kwargs']['sci_eval']['sampling']
    for i in range(len(names)):
        Xs = []
        for arr in assign:
            X_indexed = X[:, arr]
            Xs.append(X_indexed)
        name = names[i]
        debug('Evaluating {}'.format(name))
        for idx in range(len(Xs)):
            Xi = Xs[idx]
            debug('Feature {}'.format(notes[idx]))
            mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue = auc_score(config, Xi, y, name)
            text = '{},{},{},{},{:.9f},{:.9f},{:.9f},{:.9f},{},{},{},{}'.format(
                p, k, t, d, 
                mean_auc, mean_precision, mean_recall, mean_f1, 
                total_ytrue, len(y),
                notes[idx],
                name
            )
            texts.append(text)
    return texts

def sampling(X, y, ptype='original'):
    if ptype == 'original':
        return (X, y)
    ### ovesampling
    elif ptype == 'over':
        query_time = time.time()
        pp = SMOTE(kind='regular')
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling SMOTE in {} seconds'.format(process_time))
        return (X_pp, y_pp)
    ### undersampling
    elif ptype == 'under':
        query_time = time.time()
        pp = EditedNearestNeighbours()
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling ENN in {} seconds'.format(process_time))
        return (X_pp, y_pp)
    ### oversampling + undersampling
    elif ptype == 'combo':    
        query_time = time.time()
        pp = SMOTEENN()
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling SMOTE-ENN in {} seconds'.format(process_time))
        return (X_pp, y_pp)
    return (X, y)

def auc_score(config, X, y, ptype='original'):
    kfold = config['kwargs']['sci_eval']['kfold']
    n_core = config['kwargs']['n_core']
    cv = StratifiedKFold(n_splits=kfold)
    clf = RandomForestClassifier(n_jobs=n_core)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    mean_auc = 0.0

    total_ytrue = sum(y)

    i = 0
    success = 0
    for (train, test) in cv.split(X, y):
        X_pp, y_pp = sampling(X[train], y[train], ptype)
        fit = clf.fit(X_pp, y_pp)
        probas_ = fit.predict_proba(X[test])
        inference = fit.predict(X[test])
        try:
            # Compute ROC curve and area the curve
            # fpr, tpr, thresholds
            fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            mean_auc += roc_auc

            # precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
            average = 'weighted'
            precision = precision_score(y[test], inference, average=average)
            recall = recall_score(y[test], inference, average=average)
            f1 = f1_score(y[test], inference, average=average)

            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1

            success += 1
        except:
            pass
        i += 1
    mean_tpr /= success
    mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)

    mean_precision /= success
    mean_recall /= success
    mean_f1 /= success
    mean_auc /= success
    
    debug('{:.3f} {:.3f} {:.3f} {:.3f} {} {}'.format(mean_auc, mean_precision, mean_recall, mean_f1, int(total_ytrue), len(y)))

    return mean_auc, mean_precision, mean_recall, mean_f1, total_ytrue[0]