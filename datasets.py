from nltk.tokenize import wordpunct_tokenize
import csv
import os
import logging
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
        Tweet.__init__(text)
        # TODO: replace with regex. Replace wider range of emoticons.
        self.original_text = text
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


def load_mokoron_dataset_from_files(positive_filename, negative_filename, unrated_filename=None):
    parsed_dataset = []
    for filename in (positive_filename, negative_filename):
        with open(filename) as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=';')
            num = 1
            for row in reader:
                num += 1
                parsed_dataset.append(MokoronTweet.from_csv_row(row))

    if unrated_filename:
        with open(unrated_filename) as unrated_file:
            print("Loading stuff")
            for line in unrated_file:
                parsed_dataset.append(MokoronTweet.from_string(line))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_dataset)))
    # random.shuffle(parsed_dataset)
    return parsed_dataset, None


def load_mokoron_dataset_from_directory(dir_path):
    raise NotImplemented()


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


def get_dataset_stats(data):
    stats = defaultdict(int)
    for i in data:
        stats[i.polarity] += 1

    return stats


def print_dataset_stats(data, dataset_name):
    stats = get_dataset_stats(data)
    print("Dataset stats: {}".format(dataset_name))
    for polarity, count in sorted(stats.items()):
        print("{} -> {}".format(polarity, count))
