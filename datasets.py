import csv
import sys
import os
import logging
import random

from nltk.tokenize import wordpunct_tokenize
from enum import IntEnum
from io import open
from collections import defaultdict
from uuid import uuid4

_thismodule = sys.modules[__name__]

positive_emoticons_list = [":)", ":D", ")"]
negative_emoticons_list = [":(", ":C", "("]

TWEET_SENTIMENT_NEUTRAL_RANGE = (-0.33, 0.33)


class TweetSentiment(IntEnum):
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1

    @staticmethod
    def from_real_value(value, without_neutral=True):
        if without_neutral:
            return TweetSentiment.NEGATIVE if value <= 0 else TweetSentiment.POSITIVE

        neutral_low, neutral_high = TWEET_SENTIMENT_NEUTRAL_RANGE
        if value > neutral_high:
            return TweetSentiment.POSITIVE

        if value < neutral_low:
            return TweetSentiment.NEGATIVE

        return TweetSentiment.NEUTRAL


class Tweet(object):
    def __init__(self, text, polarity=None, id=None):
        self.id = id or uuid4()
        self.text = text
        self.polarity = TweetSentiment(polarity) if polarity is not None else None

    def __hash__(self):
        return hash((self.id, self.text))

    def __eq__(self, other):
        return self.id == other.id and self.text == other.text

    def is_sentiment_unknown(self):
        return self.polarity is None

    def is_neutral(self):
        return self.polarity == 0

    def is_positive(self):
        return self.polarity > 0

    def is_negative(self):
        return self.polarity < 0

    def get_text(self):
        return self.text

    def get_words(self):
        # TODO: support for different tokenizers
        return wordpunct_tokenize(self.get_text())

    @staticmethod
    def _strip_emoticons(text):
        # TODO: regex
        for emoticon in positive_emoticons_list + negative_emoticons_list:
            text = text.replace(emoticon, '')

        return text


class Sentiment140Tweet(Tweet):
    def __init__(self, row):
        text = self._strip_emoticons(row[-1])
        polarity = (int(row[0]) - 2) / 2
        assert polarity in (-1, 0, 1)

        Tweet.__init__(self, text, polarity)


class MokoronTweet(Tweet):
    def __init__(self, text, polarity):
        text = self._strip_emoticons(text)
        assert polarity in (-1, 0, 1, None)

        Tweet.__init__(self, text, polarity)

    @classmethod
    def from_string(self, s):
        return MokoronTweet(s, None)

    @classmethod
    def from_csv_row(self, row):
        # TODO: replace with regex. Replace wider range of emoticons.
        return MokoronTweet(row[3], int(row[4]))


class MyTweet(Tweet):
    def __init__(self, text, polarity):
        assert polarity in (-1, 0, 1, None)
        text = self._strip_emoticons(text)
        Tweet.__init__(self, text, polarity)


def load_dataset_by_name(dataset_name):
    if '/' in dataset_name:
        dataset_name, dataset_train_share = dataset_name.split('/')
        dataset_train_share = float(dataset_train_share)
    else:
        dataset_train_share = 1.0

    load_dataset_from_directory = getattr(_thismodule, "load_{}_from_directory".format(dataset_name))
    # dataset_dir = "datasets/{}".format(dataset_name)
    if load_dataset_from_directory is None:
        raise Exception("Unknown dataset name {}".format(dataset_name))

    dataset_train, dataset_test = load_dataset_from_directory()

    if dataset_train_share < 1.0:
        dataset_train = random.sample(dataset_train, int(len(dataset_train) * dataset_train_share))
    else:
        random.shuffle(dataset_train)

    return dataset_train, dataset_test


def load_mokoron_from_files(positive_filename, negative_filename, unlabeled_train_filename=None):
    parsed_tweets = []
    for filename in (positive_filename, negative_filename):
        with open(filename) as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=';')
            for row in reader:
                parsed_tweets.append(MokoronTweet.from_csv_row(row))

    if unlabeled_train_filename:
        with open(unlabeled_train_filename) as unrated_file:
            for line in unrated_file:
                parsed_tweets.append(MokoronTweet.from_string(line))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_tweets)))
    # random.shuffle(parsed_tweets)
    return parsed_tweets, None


def load_mokoron_from_directory(dir_path="datasets/mokoron", test_dir_path="datasets/my_test1", with_unlabeled_train=True):
    dataset_train, _ = load_mokoron_from_files(
        os.path.join(dir_path, "mokoron_positive.csv"),
        os.path.join(dir_path, "mokoron_negative.csv"),
        os.path.join(dir_path, "mokoron_unrated.txt") if with_unlabeled_train else None
    )
    dataset_test = load_my_test_from_file(os.path.join(test_dir_path, "test_dataset.csv"))

    return dataset_train, dataset_test


def load_mokoron_unrated_from_directory(dir_path="datasets/mokoron", with_unlabeled_train=True):
    dataset_train, dataset_test = load_mokoron_from_directory(dir_path, with_unlabeled_train)

    assert dataset_test is None

    rated_tweets = [tweet for tweet in dataset_train if tweet.polarity is not None]
    random.shuffle(rated_tweets)
    test_rated_tweets = rated_tweets[:1000]

    new_train = list(set(dataset_train) - set(test_rated_tweets))
    random.shuffle(new_train)

    return new_train, test_rated_tweets


def load_sentiment140_dataset_part(filename):
    parsed_file = []
    with open(filename, encoding='latin1') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            parsed_file.append(Sentiment140Tweet(row))

    return parsed_file


def load_sentiment140_dataset_from_files(train_file_name, test_file_name):
    train_data = load_sentiment140_dataset_part(train_file_name)
    test_data = load_sentiment140_dataset_part(test_file_name)

    return train_data, test_data


def load_sentiment140_from_directory(dir_path="datasets/sentiment140"):
    train_file_name = "training.1600000.processed.noemoticon.csv"
    test_file_name = "testdata.manual.2009.06.14.csv"

    train_file_path = os.path.join(dir_path, train_file_name)
    test_file_path = os.path.join(dir_path, test_file_name)

    return load_sentiment140_dataset_from_files(train_file_path, test_file_path)


def load_my_train_from_files(positive_filename, negative_filename):
    parsed_tweets = []
    for filename, polarity in ((positive_filename, 1), (negative_filename, -1)):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                parsed_tweets.append(MyTweet(row[4], polarity))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_tweets)))
    return parsed_tweets


def load_my_test_from_file(test_filename):
    parsed_tweets = []
    with open(test_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            parsed_tweets.append(MyTweet(row[6], int(row[1])))

    return parsed_tweets


def load_my_unrated_from_directory(dir_path="datasets/my_test1"):
    positive_file = "positive_my_test1.csv"
    negative_file = "negative_my_test1.csv"

    positive_file_path = os.path.join(dir_path, positive_file)
    negative_file_path = os.path.join(dir_path, negative_file)

    tweets = load_my_train_from_files(positive_file_path, negative_file_path)

    random.shuffle(tweets)

    test_size = 1000
    dataset_train = tweets[test_size:]
    dataset_test = tweets[:test_size]

    return dataset_train, dataset_test


def load_my_rated_from_directory(dir_path="datasets/my_test1"):
    positive_file = "positive_my_test1.csv"
    negative_file = "negative_my_test1.csv"
    test_file = "test_dataset.csv"

    positive_file_path = os.path.join(dir_path, positive_file)
    negative_file_path = os.path.join(dir_path, negative_file)
    test_file_path = os.path.join(dir_path, test_file)

    dataset_test = load_my_test_from_file(test_file_path)
    dataset_train = load_my_train_from_files(positive_file_path, negative_file_path)

    random.shuffle(dataset_train)

    return dataset_train, dataset_test


def load_my_eng_plus_sentiment140_from_directory(dir_path="datasets/my_eng_plus", sentiment140_dir_path="datasets/sentiment140"):
    positive_file = "my_eng_positive_head.csv"
    negative_file = "my_eng_negative_head.csv"

    positive_file_path = os.path.join(dir_path, positive_file)
    negative_file_path = os.path.join(dir_path, negative_file)

    dataset_train = load_my_train_from_files(positive_file_path, negative_file_path)
    sentiment140_train, sentiment140_test = load_sentiment140_from_directory(sentiment140_dir_path)

    dataset_train += sentiment140_train
    random.shuffle(dataset_train)

    logging.info("Loaded dataset having {} tweets".format(len(dataset_train)))
    return dataset_train, sentiment140_test


def get_dataset_stats(data):
    stats = defaultdict(int)
    for i in data:
        stats[i.polarity] += 1

    return stats


def print_dataset_stats(data, dataset_name):
    stats = get_dataset_stats(data)
    print("Dataset stats: {}".format(dataset_name))

    # None and ints are unorderable, so we have to manually delete None
    if None in stats:
        print("{} -> {}".format(None, stats[None]))
        del stats[None]

    for polarity, count in sorted(stats.items()):
        print("{} -> {}".format(polarity, count))
