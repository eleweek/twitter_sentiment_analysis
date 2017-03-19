import logging
import random

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


class TweetToFeaturesModel(object):
    pass


class SimpleDoc2Vec(TweetToFeaturesModel):
    model_name = "doc2vec"

    def save(self, filename_prefix):
        self.model.save(filename_prefix)

    @staticmethod
    def load(filename_prefix):
        new_instance = SimpleDoc2Vec()
        new_instance.model.load(filename_prefix)

        return new_instance

    @staticmethod
    def create_from_argv(*args):
        return SimpleDoc2Vec(*map(int, args))

    def __init__(self, epochs, min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=12, *args, **kwargs):
        self.model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=12, *args, **kwargs)
        self._epochs = epochs
        self._tweet_to_index = {}

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

    def infer_vector(self, tweet):
        return self.model.infer_vector(tweet.words)

    def train_vector_by_index(self, tweet):
        return self.model[SimpleDoc2Vec._train_item_tag(self._tweet_to_index(tweet))]

    @staticmethod
    def _train_item_tag(i):
        return "TRAIN_ITEM_{}".format(i)


class Fasttext(TweetToFeaturesModel):
    pass
