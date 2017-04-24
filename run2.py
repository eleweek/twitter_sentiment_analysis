#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import models
import datasets
import json
import logging

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def train_features_model(train_tweets, test_tweets, full_model_class_name, full_model_class_params, full_model_file_prefix):
    model_class = models.find_model_class_by_name(full_model_class_name)

    model = model_class(**full_model_class_params)

    model.train(train_tweets)
    model.save(full_model_file_prefix)


def train_full_model(train_tweets, test_tweets, features_model_class_name, features_model_class_params, features_model_file_prefix):
    model_class = models.find_model_class_by_name(features_model_class_name)

    model = model_class(**features_model_class_params)

    model.train(train_tweets)
    model.save(features_model_file_prefix)
    model.train(test_tweets)


def train_features_to_sentiment_model(train_tweets, test_tweets,
                                      features_model_class_name, features_model_file_prefix,
                                      features_to_sentiment_model_class_name, features_to_sentiment_model_params, features_to_sentiment_model_file_prefix):

    features_model_class = models.find_model_class_by_name(features_model_class_name)

    features_model = features_model_class.load(features_model_file_prefix)
    features_to_sentiment_model_class = models.find_model_class_by_name(features_to_sentiment_model_class_name)
    features_to_sentiment_model = features_to_sentiment_model_class(features_model, **features_to_sentiment_model_params)

    features_to_sentiment_model.train(train_tweets)
    features_to_sentiment_model.save(features_to_sentiment_model_file_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tweet_to_features_model',
                        default=False,
                        action='store_true',
                        help="Launch training tweet_to_features model(e.g. word2vec/doc2vec)")
    parser.add_argument('--train_features_to_sentiment_model',
                        default=False,
                        action='store_true',
                        help="Launch training features_to_sentiment model(e.g. from embeddings to sentiment)")
    parser.add_argument('--train_full_model',
                        default=False,
                        action='store_true',
                        help='Launch training model from tweets to sentiment. E.g. keras models with embedding layer.')

    parser.add_argument('--test_features_to_sentiment',
                        default=False,
                        action='store_true',
                        help="Launch testing features_to_sentiment model(e.g. from embeddings to sentiment)")
    parser.add_argument('--test_full_model',
                        default=False,
                        action='store_true',
                        help='Launch testing from tweets to sentiment. E.g. keras models with embedding layer.')

    parser.add_argument('--tweet_to_features_model',
                        help="For training tweet_to_features(e.g. embedding model like doc2vec, fasttext). Specify model name here")
    parser.add_argument('--tweet_to_features_model_file_prefix',
                        help="File prefix for saving model")
    parser.add_argument('--tweet_to_features_model_params',
                        default='{}',
                        help="JSON with kwargs for model",
                        type=json.loads)

    parser.add_argument('--features_to_sentiment_model',
                        help="Features_to_sentiment model class name")
    parser.add_argument('--features_to_sentiment_model_file_prefix',
                        help="File prefix for saving model")
    parser.add_argument('--features_to_sentiment_model_params',
                        default='{}',
                        help="JSON with kwargs for model",
                        type=json.loads)

    parser.add_argument('--full_model',
                        help="Full model class name")
    parser.add_argument('--full_model_file_prefix',
                        help="File prefix for saving model")
    parser.add_argument('--full_model_params',
                        default='{}',
                        help="JSON with kwargs for model",
                        type=json.loads)

    parser.add_argument('--dataset',
                        help="Dataset name",
                        required=True)

    args = parser.parse_args()

    features_model_class_name = args.tweet_to_features_model
    features_to_sentiment_model_class_name = args.features_to_sentiment_model
    full_model_class_name = args.full_model

    dataset_name = args.dataset
    features_model_file_prefix = args.tweet_to_features_model_file_prefix
    features_to_sentiment_model_file_prefix = args.features_to_sentiment_model_file_prefix
    full_model_file_prefix = args.full_model_file_prefix

    features_model_params = args.tweet_to_features_model_params
    features_to_sentiment_model_params = args.tweet_to_features_model_params
    full_model_params = args.full_model_params

    train_tweets, test_tweets = datasets.load_dataset_by_name(dataset_name)

    if args.train_tweet_to_features_model:
        train_features_model(train_tweets, features_model_class_name, features_model_params, features_model_file_prefix)
    elif args.train_features_to_sentiment_model:
        train_features_to_sentiment_model(train_tweets, test_tweets,
                                          features_model_class_name, features_model_file_prefix,
                                          features_to_sentiment_model_class_name, features_to_sentiment_model_params, features_to_sentiment_model_file_prefix)
    elif args.train_full_model:
        train_full_model(train_tweets, test_tweets, full_model_class_name, full_model_params, full_model_file_prefix)


if __name__ == "__main__":
    main()
