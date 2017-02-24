import logging
import csv
import sys
import random
from collections import defaultdict

import numpy
from nltk.tokenize import wordpunct_tokenize
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from io import open

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class Sentiment140Tweet(object):
    def __init__(self, row):
        self.original_text = row[-1]
        self.polarity = (int(row[0]) - 2) / 2
        self.words = wordpunct_tokenize(self.original_text)

    def get_words(self):
        # TODO: support for different tokenizers
        return self.words

    def is_neutral(self):
        return self.polarity == 0

    def is_positive(self):
        return self.polarity > 0

    def is_negative(self):
        return self.polarity < 0


def load_data_from_csv(filename):
    parsed_file = []
    with open(filename, encoding='latin1') as csvfile:
        reader = csv.reader(csvfile)
        num = 0
        for row in reader:
            num += 1
            parsed_file.append(Sentiment140Tweet(row))

    return parsed_file


def run(train_file_csv, test_file_csv, model_file_name):
    train_file = load_data_from_csv(train_file_csv)
    test_file = load_data_from_csv(test_file_csv)

    test_file = [tweet for tweet in test_file if not tweet.is_neutral()]
    logging.info("Len test file = {}".format(len(test_file)))

    model = Doc2Vec.load(model_file_name)
    train_arrays = numpy.zeros((len(train_file), 100))
    train_labels = numpy.zeros(len(train_file))

    test_arrays = numpy.zeros((len(test_file), 100))
    test_labels = numpy.zeros(len(test_file))

    for i in range(len(train_file)):
        prefix_train_pos = 'TRAIN_ITEM_{}'.format(i)

        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_labels[i] = train_file[i].polarity

    for i, tweet in enumerate(test_file):
        test_arrays[i] = model.infer_vector(tweet.words)
        # if polarity == 2:
        #    continue
        # test_arrays[i] = model[text]
        # test_labels[i] = polarity / 4
        # test_labels[i] = 2 * test_labels[i] - 1
        test_labels[i] = tweet.polarity

    logging.info('Fitting')
    # clf = LogisticRegression()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
    # clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
    # classifier.fit(train_arrays, train_labels)
    # clf = svm.SVC()
    # clf = KNeighborsClassifier(5)
    # clf = RandomForestClassifier(max_depth=5, n_estimators=500)
    clf.fit(train_arrays, train_labels)

    logging.info("Done fitting")
    print(clf.score(train_arrays, train_labels))
    print(clf.score(test_arrays, test_labels))


def train_doc2vec(train_file_csv, model_file_name, epochs):
    train_file = load_data_from_csv(train_file_csv)

    tagged_docs = []
    for index, tweet in enumerate(train_file):
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
    if sys.argv[1] == "train_doc2vec":
        train_doc2vec(sys.argv[2], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "train_and_test_classifier":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        assert False
