#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import models
import datasets
import inspect
import logging
import random
import numpy as np
from sklearn.neural_network import MLPClassifier

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def train_features_model(dataset_name, model_class_name, model_file_name, *args):
    model_class = find_model_class_by_name(model_class_name)

    model = model_class.create_from_argv(*args)
    dataset_train, dataset_test = load_dataset_by_name(dataset_name)

    model.train(dataset_train)
    model.save(model_file_name)


def convert_dataset_to_features(tweets, model):
    logging.debug("convert_dataset_to_features: number of features is {}".format(model.get_features_number()))
    labels = np.zeros(len(tweets))

    for i, tweet in enumerate(tweets):
        labels[i] = tweet.polarity
        assert tweet.polarity is not None and -1.0 <= tweet.polarity <= 1.0

    if hasattr(model, "batch_get_features"):
        arrays = model.batch_get_features(tweets)
        assert len(arrays) == len(tweets)
    else:
        arrays = np.zeros((len(tweets), model.get_features_number()))
        for i, tweet in enumerate(tweets):
            arrays[i] = model.get_features(tweet)

    return arrays, labels


def test_features_model(dataset_name, model_class_name, model_file_name, *args):
    model_class = find_model_class_by_name(model_class_name)
    model = model_class.load(model_file_name)

    dataset_train, dataset_test = load_dataset_by_name(dataset_name)

    def remove_unrated(tweets):
        return [tweet for tweet in tweets if tweet.polarity is not None]

    dataset_test = remove_unrated([tweet for tweet in dataset_test if not tweet.is_neutral()])
    dataset_train = remove_unrated(dataset_train)

    datasets.print_dataset_stats(dataset_train, "{}:train".format(dataset_name))
    datasets.print_dataset_stats(dataset_test, "{}:test".format(dataset_name))

    train_arrays, train_labels = convert_dataset_to_features(dataset_train, model)
    print(np.where(np.isnan(train_arrays)))
    print(np.where(np.isinf(train_arrays)))
    test_arrays, test_labels = convert_dataset_to_features(dataset_test, model)

    logging.info("Starting fitting")
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, ), random_state=1)
    clf.fit(train_arrays, train_labels)

    logging.info("Done fitting")
    print(clf.score(train_arrays, train_labels))
    print(clf.score(test_arrays, test_labels))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "train_features_model":
        train_features_model(*sys.argv[2:])
    elif action == "test_features_model":
        test_features_model(*sys.argv[2:])
    else:
        raise Exception("Unknown action {}".format(action))
