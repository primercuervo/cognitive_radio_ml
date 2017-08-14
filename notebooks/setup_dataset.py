#!/usr/bin/env python

'''
sets up the dataset based on the measurement data

Argumets: relative path

'''
# Required modules for all notebooks
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import IPython
import sklearn
import graphviz
import sys
import os
# Import the constants
import constants as c
import functions as fun

# Common used functions from sklearn
from sklearn.metrics import accuracy_score
from time import time

# Create iterables for the features
# Interframe time per channel per scenario
def setup_iterables(path):
    """
    Sets up the iterables to work with and populates them with the data located
    at *path*
    """
    abs_path = os.path.join(c.PRE_PATH, path)
    if_time_scn_ch = [[[channel] for channel in range(c.N_CHAN)]
                      for scenario in range(c.N_SCN)]

    # Packet rate per scenario
    packet_rate_scn = [[] for scenario in range(c.N_SCN)]

    # Variance per scenario
    variance_scn = [[] for scenario in range(c.N_SCN)]

    # Generate a list that includes the interframe time for all channels
    if_vector = [[] for i in range(c.N_SAMPS * c.N_SCN)]
    try:
        for scenario in range(c.N_SCN):
            for channel in range(c.N_CHAN):
                if_time_scn_ch[scenario][channel] = sp.fromfile(open(
                    os.path.join(abs_path, "interframe_time_ch_{}_scn_{}.dat"
                                 .format(channel+1, scenario))), dtype=sp.float32)
            packet_rate_scn[scenario] = sp.fromfile(open(
                os.path.join(abs_path, "packet_rate_scn_{}.dat"
                             .format(scenario))), dtype=sp.float32)
            variance_scn[scenario] = sp.fromfile(open(
                os.path.join(abs_path, "variance_scn_{}.dat"
                             .format(scenario))), dtype=sp.float32)
    except IOError as e:
        print("Error trying to access path: ", e)
        raise

    # Populate the conglomerated interframe_time list
    for scn in range(c.N_SCN):
        for i in range(c.N_SAMPS):
            for chan in range(c.N_CHAN):
                if_vector[i + c.N_SAMPS*scn].append(if_time_scn_ch[scn][chan][i])

    # Generate label list
    labels = [i for i in range(c.N_SCN) for n in range(c.N_SAMPS)]

    # Generate a list that includes all data in a list per frames
    data_nested = []
    # first generate a long list that includes the packet_rates one scenario
    # after the other, and the same for the variances
    # packet_rate = [scn0, scn1, ..., scn9]
    # len(packet_rate) = N_SAMPS * N_SCN
    packet_rate = []
    variance = []
    for scn in range(c.N_SCN):
        for i in range(c.N_SAMPS):
            packet_rate.append(packet_rate_scn[scn][i])
            variance.append(variance_scn[scn][i])

    data_nested = list(zip(if_vector, packet_rate, variance))
    # Until this point 'data' is a nested list. It needs to be flattened
    # to use it with sci-kit
    # TODO: just don't generate it nested and save this method...
    data = [[] for i in range(len(data_nested))]
    for i in range(len(data_nested)):
        data[i] = list(fun.flatten(data_nested[i]))
    return data, labels

def slice_data(data, labels):
    """
    Takes the dataset and divides it N_SLICES. This is to check the effect that
    the dataset size has in the accuracy of the models
    IN: Two lists: data and labels
    OUT: Four lists of size N_SLICES, each with sliced and randomized data
    with different sizes.

    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = [[[] for _ in range(c.NUM_SLICES)] for _ in range(4)]
    data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                        labels,
                                                                        random_state=42)

    for n in range(c.NUM_SLICES, 0, -1):
        X_train[c.NUM_SLICES-n] = data_train[:int(len(data_train)/n)]
        X_test[c.NUM_SLICES-n] = data_test[:int(len(data_test)/n)]
        y_train[c.NUM_SLICES-n] = labels_train[:int(len(labels_train)/n)]
        y_test[c.NUM_SLICES-n] = labels_test[:int(len(labels_test)/n)]
    return X_train, X_test, y_train, y_test

def scale_sliced_data(X_train, X_test, scaler):
    """
    Takes features data and scales it based on the selected scaler given in the
    arguments.
    CAUTION: The given scaler has to exists in sklearn.preprocessing
    """
    X_train_scaled, X_test_scaled = [[[] for _ in range(c.NUM_SLICES)] for _ in range(2)]
    scaler_ = scaler
    for i in range(c.NUM_SLICES):
        X_train_scaled[i] = scaler_.fit_transform(X_train[i])
        X_test_scaled[i] = scaler_.transform(X_test[i])
    return X_train_scaled, X_test_scaled

def run_knn(X_train, X_test, y_train, y_test, n_neighbors):
    """
    runs K-nearest neighbors with a set of complexities over the sliced data
    and plots a whole bunch of results
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn_list, knn_predictions, knn_accs, knn_pred_times, knn_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for n in range(len(n_neighbors)):
            knn_list[i].append(KNeighborsClassifier(n_neighbors=n_neighbors[n]))
            t0 = time()
            knn_list[i][n].fit(X_train[i], y_train[i])
            knn_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            knn_predictions[i].append(knn_list[i][n].predict(X_test[i]))
            knn_pred_times.append(round(time() - t0, 3))
            knn_accs[i].append(accuracy_score(y_test[i], knn_predictions[i][n]))
    return knn_accs, knn_predictions, knn_pred_times, knn_fit_times

def run_svc(X_train, X_test, y_train, y_test, complexities):
    """
    Runs Support Vector Machine classifier for the given data

    Returns:
    - list of accuracies for different dataset size given and complexities
    - list of fit times
    - list of prediction times
    - list of predictions
    """
    from sklearn.svm import SVC
    svc_list, svc_pred, svc_accs, svc_pred_times, svc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]

    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            svc_list[i].append(SVC(kernel='rbf', C=float(complexities[n])))
            t0 = time()
            svc_list[i][n].fit(X_train[i], y_train[i])
            svc_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            svc_pred[i].append(svc_list[i][n].predict(X_test[i]))
            svc_pred_times.append(round(time() - t0, 3))
            svc_accs[i].append(accuracy_score(y_test[i], svc_pred[i][n]))
    return svc_accs, svc_pred, svc_pred_times, svc_fit_times

def run_random_forest(X_train, X_test, y_train, y_test, complexities, jobs):
    """
    Run a random forest classification over the dataset
    The complexity here is ruled by the number of estimators that the model takes
    In addition, it takes the arg "jobs", as this model allows parallelization
    """
    from sklearn.ensemble import RandomForestClassifier
    rfc_list, rfc_pred, rfc_accs, rfc_pred_times, rfc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for job in range(len(jobs)):
            for n in range(len(complexities)):
                rfc_list[i].append(RandomForestClassifier(n_estimators=complexities[n],
                                                          n_jobs=jobs[job]))
                t0 = time()
                rfc_list[i][n].fit(X_train[i], y_train[i])
                rfc_fit_times[i].append(round(time() - t0, 3))
                t0 = time()
                rfc_pred[i].append(rfc_list[i][n].predict(X_test[i]))
                rfc_pred_times.append(round(time() - t0, 3))
                rfc_accs[i].append(accuracy_score(y_test[i], rfc_pred[i][n]))
    return rfc_accs, rfc_pred, rfc_pred_times, rfc_fit_times

def run_gaussian(X_train, X_test, y_train, y_test, complexities, jobs):
    """
    Run a Gaussian Process classification over the dataset
    The complexity here is ruled by the number of iterations the optimization
    algorithm does (Newton descent)
    In addition, it takes the arg "jobs", as this model allows parallelization
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    gpc_list, gpc_pred, gpc_accs, gpc_pred_times, gpc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for job in range(len(jobs)):
            for n in range(len(complexities)):
                gpc_list[i].append(GaussianProcessClassifier(kernel=1.0 * RBF(1.0),
                                                             n_jobs=jobs[job],
                                                             warm_start=True))
                t0 = time()
                gpc_list[i][n].fit(X_train[i], y_train[i])
                gpc_fit_times[i].append(round(time() - t0, 3))
                t0 = time()
                gpc_pred[i].append(gpc_list[i][n].predict(X_test[i]))
                gpc_pred_times.append(round(time() - t0, 3))
                gpc_accs[i].append(accuracy_score(y_test[i], gpc_pred[i][n]))
    return gpc_accs, gpc_pred, gpc_pred_times, gpc_fit_times

def run_decision_tree(X_train, X_test, y_train, y_test, complexities):
    """
    Run a decision tree classification over the dataset
    The complexity here is ruled by the number of samples considered for every split
    """
    from sklearn.tree import DecisionTreeClassifier

    dtc_list, dtc_pred, dtc_accs, dtc_pred_times, dtc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            dtc_list[i].append(DecisionTreeClassifier(max_depth=complexities[n]))
            t0 = time()
            dtc_list[i][n].fit(X_train[i], y_train[i])
            dtc_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            dtc_pred[i].append(dtc_list[i][n].predict(X_test[i]))
            dtc_pred_times.append(round(time() - t0, 3))
            dtc_accs[i].append(accuracy_score(y_test[i], dtc_pred[i][n]))
    return dtc_accs, dtc_pred, dtc_pred_times, dtc_fit_times

def run_naive_bayes(X_train, X_test, y_train, y_test, complexities):
    """
    Run a Naive Bayes classification over the dataset
    No complexities applied
    """
    from sklearn.naive_bayes import GaussianNB

    nbc_list, nbc_pred, nbc_accs, nbc_pred_times, nbc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            nbc_list[i].append(GaussianNB())
            t0 = time()
            nbc_list[i][n].fit(X_train[i], y_train[i])
            nbc_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            nbc_pred[i].append(nbc_list[i][n].predict(X_test[i]))
            nbc_pred_times.append(round(time() - t0, 3))
            nbc_accs[i].append(accuracy_score(y_test[i], nbc_pred[i][n]))
    return nbc_accs, nbc_pred, nbc_pred_times, nbc_fit_times

def run_adaboost(X_train, X_test, y_train, y_test, complexities):
    """
    Run a AdaBoost Classifier classification over the dataset
    No complexities applied
    """
    from sklearn.ensemble import AdaBoostClassifier

    abc_list, abc_pred, abc_accs, abc_pred_times, abc_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            abc_list[i].append(AdaBoostClassifier())
            t0 = time()
            abc_list[i][n].fit(X_train[i], y_train[i])
            abc_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            abc_pred[i].append(abc_list[i][n].predict(X_test[i]))
            abc_pred_times.append(round(time() - t0, 3))
            abc_accs[i].append(accuracy_score(y_test[i], abc_pred[i][n]))
    return abc_accs, abc_pred, abc_pred_times, abc_fit_times

def run_quadratic(X_train, X_test, y_train, y_test, complexities):
    """
    Run a Quadratic Discriminator analysis Classifier classification over the dataset
    No complexities applied
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    qda_list, qda_pred, qda_accs, qda_pred_times, qda_fit_times =\
            [[[] for _ in range(c.NUM_SLICES)] for _ in range(5)]
    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            qda_list[i].append(QuadraticDiscriminantAnalysis())
            t0 = time()
            qda_list[i][n].fit(X_train[i], y_train[i])
            qda_fit_times[i].append(round(time() - t0, 3))
            t0 = time()
            qda_pred[i].append(qda_list[i][n].predict(X_test[i]))
            qda_pred_times.append(round(time() - t0, 3))
            qda_accs[i].append(accuracy_score(y_test[i], qda_pred[i][n]))
    return qda_accs, qda_pred, qda_pred_times, qda_fit_times
def compute_cm(y_test, predictions, complexities, normalized=False, verbose=False):
    """
    computes and plots the confusion matrices from the predictions
    """
    from sklearn.metrics import confusion_matrix

    cnf_matrix = [[] for _ in range(c.NUM_SLICES)]
    for i in range(c.NUM_SLICES):
        for n in range(len(complexities)):
            cnf_matrix[i].append(confusion_matrix(y_test[i], predictions[i][n]))
            np.set_printoptions(precision=2)

            plt.figure(figsize=(10, 10))
            fun.plot_confusion_matrix(cnf_matrix[i][n],
                                      classes=c.CLASS_NAMES,
                                      verbose=verbose,
                                      title='Confusion Matrix')
            # TODO: set up a way to show the confusion matrix title descriptive
    plt.show()

