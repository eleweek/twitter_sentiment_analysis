import logging
import sys
import random

import numpy
from sklearn.neural_network import MLPClassifier

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def sentiment140_train_and_test_classifier(train_file_csv, test_file_csv, model_file_name):
    train_data = load_sentiment140_dataset_part(train_file_csv)
    test_data = load_sentiment140_dataset_part(test_file_csv)
    test_data = [tweet for tweet in test_data if not tweet.is_neutral()]

    doc2vec_model = Doc2Vec.load(model_file_name)
    _train_and_test_classifier(train_data, test_data, doc2vec_model)


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
