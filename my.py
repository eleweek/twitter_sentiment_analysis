import logging
import csv
import sys
import random
from collections import defaultdict

import numpy
from nltk.tokenize import wordpunct_tokenize
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.neural_network import MLPClassifier

from io import open

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class Tweet(object):
    def is_neutral(self):
        return self.polarity == 0

    def is_positive(self):
        return self.polarity > 0

    def is_negative(self):
        return self.polarity < 0

    def get_words(self):
        # TODO: support for different tokenizers
        return self.words


class Sentiment140Tweet(Tweet):
    def __init__(self, row):
        self.original_text = row[-1]
        self.text = row[-1]
        self.polarity = (int(row[0]) - 2) / 2
        assert self.polarity in (-1, 0, 1)
        self.words = wordpunct_tokenize(self.text)


class MokoronTweet(Tweet):
    def __init__(self, text, polarity=None):
        # TODO: replace with regex. Replace wider range of emoticons.
        self.original_text = text
        self.original_text = self.original_text.replace(':)', '').replace(':(', '').replace(':D', '').replace(')', '').replace('(', '')
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


def load_and_shuffle_mokoron_dataset(positive_filename, negative_filename, unrated_filename=None):
    parsed_dataset = []
    for filename in (positive_filename, negative_filename):
        with open(filename) as csvfile:
            reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=';')
            num = 1
            for row in reader:
                # print(num, row)
                num += 1
                parsed_dataset.append(MokoronTweet.from_csv_row(row))
                # print(parsed_dataset[-1].words, parsed_dataset[-1].polarity)

    if unrated_filename:
        with open(unrated_filename) as unrated_file:
            print("Loading stuff")
            for line in unrated_file:
                parsed_dataset.append(MokoronTweet.from_string(line))

    logging.info("Loaded dataset having {} tweets".format(len(parsed_dataset)))
    # random.shuffle(parsed_dataset)
    return parsed_dataset


def load_sentiment140_dataset_part(filename):
    parsed_file = []
    with open(filename, encoding='latin1') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            parsed_file.append(Sentiment140Tweet(row))

    return parsed_file


def sentiment140_train_and_test_classifier(train_file_csv, test_file_csv, model_file_name):
    train_data = load_sentiment140_dataset_part(train_file_csv)
    test_data = load_sentiment140_dataset_part(test_file_csv)
    test_data = [tweet for tweet in test_data if not tweet.is_neutral()]

    doc2vec_model = Doc2Vec.load(model_file_name)
    _train_and_test_classifier(train_data, test_data, doc2vec_model)


def print_dataset_stats(data):
    hist = defaultdict(int)
    for i in data:
        hist[i.polarity] += 1

    for polarity, count in sorted(hist.items()):
        print(polarity, count)


def _train_and_test_classifier(train_data, test_data, doc2vec_model):
    logging.info("Len test data = {}".format(len(test_data)))
    print_dataset_stats(train_data)
    print_dataset_stats(test_data)

    train_arrays = numpy.zeros((len(train_data), 100))
    train_labels = numpy.zeros(len(train_data))

    test_arrays = numpy.zeros((len(test_data), 100))
    test_labels = numpy.zeros(len(test_data))

    for i in range(len(train_data)):
        prefix_train_pos = 'TRAIN_ITEM_{}'.format(i)

        train_arrays[i] = doc2vec_model.docvecs[prefix_train_pos]
        train_labels[i] = train_data[i].polarity

    for i, tweet in enumerate(test_data):
        test_arrays[i] = doc2vec_model.infer_vector(tweet.words)
        test_labels[i] = tweet.polarity

    logging.info('Fitting')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
    clf.fit(train_arrays, train_labels)

    logging.info("Done fitting")
    print(clf.score(train_arrays, train_labels))
    print(clf.score(test_arrays, test_labels))


def mokoron_train_and_test_classifier(train_file_prefix, model_file_name):
    train_data = load_and_shuffle_mokoron_dataset(train_file_prefix + "positive.csv", train_file_prefix + "negative.csv")
    doc2vec_model = Doc2Vec.load(model_file_name)

    _train_and_test_classifier(train_data, train_data, doc2vec_model)


def train_sentiment140_doc2vec(train_file_csv, model_file_name, epochs):
    train_data = load_sentiment140_dataset_part(train_file_csv)
    train_doc2vec(train_data, model_file_name, epochs)


def train_mokoron_doc2vec(train_file_prefix, model_file_name, epochs):
    train_data = load_and_shuffle_mokoron_dataset(train_file_prefix + "positive.csv", train_file_prefix + "negative.csv", train_file_prefix + "unrated.txt")

    train_doc2vec(train_data, model_file_name, epochs)


def train_doc2vec(train_data, model_file_name, epochs):
    tagged_docs = []
    for index, tweet in enumerate(train_data):
        tagged_docs.append(TaggedDocument(tweet.words, ["TRAIN_ITEM_{}".format(index)]))

    logging.info('D2V')
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=12)
    model.build_vocab(tagged_docs)

    logging.info('Epoch')
    for epoch in range(epochs):
        logging.info('EPOCH: {}'.format(epoch))
        random.shuffle(tagged_docs)
        model.train(tagged_docs)

    model.save(model_file_name)


if __name__ == "__main__":
    if sys.argv[1] == "sentiment140_train_doc2vec":
        train_sentiment140_doc2vec(sys.argv[2], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "sentiment140_train_and_test_classifier":
        sentiment140_train_and_test_classifier(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "mokoron_train_doc2vec":
        train_mokoron_doc2vec(sys.argv[2], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "mokoron_train_and_test_classifier":
        mokoron_train_and_test_classifier(sys.argv[2], sys.argv[4])
    else:
        assert False
