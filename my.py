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

def load_data_from_csv(filename):
    parsed_file = []
    with open(filename, encoding='latin1') as csvfile:
        reader = csv.reader(csvfile)
        num = 0
        for row in reader:
            num += 1
            polarity = int(row[0])
            text = row[-1]
            # print(polarity, ' '.join(wordpunct_tokenize(text)))
            words = wordpunct_tokenize(text)
            parsed_file.append((polarity, words))

    return parsed_file


"""
def get_np_vectors(model, data, is_train):
    n_polarized = sum([1 for d in data if d[0] != 2])

    arrays = numpy.zeros((n_polarized, 100))
    labels = numpy.zeros(n_polarized)

    for enumerate(d
"""


def run(train_file_csv, test_file_csv, model_file_name):
    train_file = load_data_from_csv(train_file_csv)
    test_file = load_data_from_csv(test_file_csv)

    # test_file = [(polarity, text) for (polarity, text) in test_file if polarity != 2]
    test_file = [(polarity, text) for (polarity, text) in test_file]
    logging.info("Len test file = {}".format(len(test_file)))

    model = Doc2Vec.load(model_file_name)
    train_arrays = numpy.zeros((len(train_file), 100))
    train_labels = numpy.zeros(len(train_file))

    test_arrays = numpy.zeros((len(test_file), 100))
    test_labels = numpy.zeros(len(test_file))

    for i in range(len(train_file)):
        prefix_train_pos = 'TRAIN_ITEM_{}'.format(i)

        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_labels[i] = train_file[i][0] / 4
        train_labels[i] = 2 * train_labels[i] - 1

    for i, (polarity, text) in enumerate(test_file):
        test_arrays[i] = model.infer_vector(text)
        # if polarity == 2:
        #    continue
        # test_arrays[i] = model[text]
        # test_labels[i] = polarity / 4
        # test_labels[i] = 2 * test_labels[i] - 1
        test_labels[i] = (polarity - 2) / 2


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
    """
    pred = [[0] * 3 for _ in range(3)]
    for p, label in zip(clf.predict(test_arrays), test_labels):
        # p = int(p)
        label = int(label)
        # print(label, p)
        if p >= 0.35:
            p = 1
        elif p <= -0.35:
            p = -1
        else:
            p = 0
        pred[label + 1][p + 1] += 1

    for label in range(3):
        print(' '.join(map(str, pred[label])))
    """


def train(train_file_csv, test_file_csv, model_file_name, epochs):
    train_file = load_data_from_csv(train_file_csv)
    test_file = load_data_from_csv(test_file_csv)

    tagged_docs = []
    for index, (polarity, text) in enumerate(train_file):
        tagged_docs.append(TaggedDocument(text, ["TRAIN_ITEM_{}".format(index)]))

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
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        assert False
