from nltk.tokenize import wordpunct_tokenize
import csv
import os
import logging
import random
from io import open
from collections import defaultdict


positive_emoticons_list = [":)", ":D", ")"]
negative_emoticons_list = [":(", ":C", "("]


class Tweet(object):
    def __init__(self, author=None):
        self.author = author
        self.id = None

    def __hash__(self):
        return hash((self.author, self.id, self.original_text))

    def __eq__(self, other):
        return self.author == other.author and self.id == other.id and self.words == other.words

    def is_neutral(self):
        return self.polarity == 0

    def is_positive(self):
        return self.polarity > 0

    def is_negative(self):
        return self.polarity < 0

    def get_words(self):
        # TODO: support for different tokenizers
        return self.words

    def get_text(self):
        return self.text

    @staticmethod
    def _strip_emoticons(text):
        # TODO: regex
        for emoticon in positive_emoticons_list + negative_emoticons_list:
            text = text.replace(emoticon, '')

        return text


class Sentiment140Tweet(Tweet):
    def __init__(self, row):
        Tweet.__init__(self)
        self.original_text = row[-1]
        self.text = row[-1]
        self.polarity = (int(row[0]) - 2) / 2
        assert self.polarity in (-1, 0, 1)
        self.words = wordpunct_tokenize(self.text)


class MokoronTweet(Tweet):
    def __init__(self, text, polarity=None):
        Tweet.__init__(self)
        # TODO: replace with regex. Replace wider range of emoticons.
        self.original_text = self._strip_emoticons(self.original_text)
        self.polarity = polarity
        assert self.polarity in (-1, 0, 1, None)
        self.words = wordpunct_tokenize(self.original_text)

    @classmethod
    def from_string(self, s):
        return MokoronTweet(s)

    @classmethod
    def from_csv_row(self, row):
        # TODO: replace with regex. Replace wider range of emoticons.
        return MokoronTweet(row[3], int(row[4]))


class MyTweet(Tweet):
    def __init__(self, text, polarity):
        Tweet.__init__(self)
        self.original_text = self._strip_emoticons(text)

        self.polarity = polarity
        assert self.polarity in (-1, 0, 1, None)
        self.words = wordpunct_tokenize(self.original_text)


def load_mokoron_from_files(positive_filename, negative_filename, unrated_filename=None):
    parsed_tweets = []
    for filename in (positive_filename, negative_filename):
        with open(filename) as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=';')
            for row in reader:
                parsed_tweets.append(MokoronTweet.from_csv_row(row))

    if unrated_filename:
        with open(unrated_filename) as unrated_file:
            for line in unrated_file:
                parsed_tweets.append(MokoronTweet.from_string(line))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_tweets)))
    # random.shuffle(parsed_tweets)
    return parsed_tweets, None


def load_mokoron_from_directory(dir_path="datasets/mokoron", with_unrated=True):
    return load_mokoron_from_files(
        os.path.join(dir_path, "mokoron_positive.csv"),
        os.path.join(dir_path, "mokoron_negative.csv"),
        os.path.join(dir_path, "mokoron_unrated.txt")
    )


def load_mokoron_shuffle_split_from_directory(dir_path="datasets/mokoron", with_unrated=True):
    dataset_train, dataset_test = load_mokoron_from_directory(dir_path)

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


def load_my_from_files(positive_filename, negative_filename):
    parsed_tweets = []
    for filename, polarity in ((positive_filename, 1), (negative_filename, -1)):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                parsed_tweets.append(MyTweet(row[4], polarity))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_tweets)))
    return parsed_tweets


def load_my_from_directory(dir_path="datasets/my_test1"):
    positive_file = "positive_my_test1.csv"
    negative_file = "negative_my_test1.csv"

    positive_file_path = os.path.join(dir_path, positive_file)
    negative_file_path = os.path.join(dir_path, negative_file)

    tweets = load_my_from_files(positive_file_path, negative_file_path)

    random.shuffle(tweets)

    test_size = 1000
    dataset_train = tweets[test_size:]
    dataset_test = tweets[:test_size]

    return dataset_train, dataset_test


def get_dataset_stats(data):
    stats = defaultdict(int)
    for i in data:
        stats[i.polarity] += 1

    return stats


def print_dataset_stats(data, dataset_name):
    stats = get_dataset_stats(data)
    print("Dataset stats: {}".format(dataset_name))
    if None in stats:
        print("{} -> {}".format(None, stats[None]))
        del stats[None]
    # None and ints are unorderable, so we have to manually delete None
    for polarity, count in sorted(stats.items()):
        print("{} -> {}".format(polarity, count))
