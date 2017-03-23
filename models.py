import logging
import random

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import fasttext


class TweetToFeaturesModel(object):
    @staticmethod
    def _check_features_range(features, l=-1.5, r=1.5):
        if not all(l <= f <= r for f in features):
            logging.warning("Feature outside of {} .. {} range: {}".format(l, r, features))


class Fasttext(TweetToFeaturesModel):
    model_name = "fasttext"

    @staticmethod
    def load(file_name):
        instance = Fasttext()
        instance.model = fasttext.load_model(file_name)

        return instance

    def train(self, train_data):
        assert NotImplementedError("Fasttext currently supports only pre-trained models. You can find them here: "
                                   "https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md")

    def get_features(self, tweet):
        features = self.model[' '.join(tweet.words)]
        self._check_features_range(features, -100, 100)
        return features

    def get_features_number(self):
        return 300  # since only pre-trained models are supported


class SimpleDoc2Vec(TweetToFeaturesModel):
    model_name = "doc2vec"

    def save(self, filename_prefix):
        self.model.save(filename_prefix)

    @staticmethod
    def load(filename_prefix):
        new_instance = SimpleDoc2Vec(None)
        new_instance.model = Doc2Vec.load(filename_prefix)

        return new_instance

    @staticmethod
    def create_from_argv(*args):
        return SimpleDoc2Vec(*map(int, args))

    def __init__(self, epochs, min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=12, *args, **kwargs):
        self.model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=12, *args, **kwargs)
        self._epochs = epochs
        self._tweet_to_index = {}
        self.features_number = size

    def train(self, train_data):
        logging.info("Training Doc2Vec model")
        epochs = self._epochs

        tagged_docs = []
        for index, tweet in enumerate(train_data):
            tagged_docs.append(TaggedDocument(tweet.words, [self._train_item_tag(index)]))
            self._tweet_to_index[tweet] = index

        self.model.build_vocab(tagged_docs)

        for epoch in range(epochs):
            logging.info('Training Doc2Vec: EPOCH: {}'.format(epoch))
            random.shuffle(tagged_docs)
            self.model.train(tagged_docs)

    def get_features(self, tweet):
        features = self.model.infer_vector(tweet.words)
        self._check_features_range(features)
        return features

    def train_vector_by_index(self, tweet):
        return self.model[SimpleDoc2Vec._train_item_tag(self._tweet_to_index(tweet))]

    def get_features_number(self):
        return self.features_number

    @staticmethod
    def _train_item_tag(i):
        return "TRAIN_ITEM_{}".format(i)
