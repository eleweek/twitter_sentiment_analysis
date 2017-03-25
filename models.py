import logging
import random
import re

from collections import OrderedDict

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import fasttext


class TweetToFeaturesModel(object):
    @staticmethod
    def _check_features_range(features, l=-2.5, r=2.5):
        if not all(l <= f <= r for f in features):
            logging.warning("Feature outside of {} .. {} range: {}".format(l, r, features))


class Fasttext(TweetToFeaturesModel):
    """
    Represents class for Facebook's fasttext models.
    Pre-trained models can be found here:
    https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    """
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
        features = self.model[tweet.get_text()]
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
            tagged_docs.append(TaggedDocument(tweet.get_words(), [self._train_item_tag(index)]))
            self._tweet_to_index[tweet] = index

        self.model.build_vocab(tagged_docs)

        for epoch in range(epochs):
            logging.info('Training Doc2Vec: EPOCH: {}'.format(epoch))
            random.shuffle(tagged_docs)
            self.model.train(tagged_docs)

    def get_features(self, tweet):
        features = self.model.infer_vector(tweet.get_words())
        self._check_features_range(features)
        return features

    def train_vector_by_index(self, tweet):
        return self.model[SimpleDoc2Vec._train_item_tag(self._tweet_to_index(tweet))]

    def get_features_number(self):
        return self.features_number

    @staticmethod
    def _train_item_tag(i):
        return "TRAIN_ITEM_{}".format(i)


class RussianSentimentLexicon(object):
    """
    Represents a class for sentiment lexicon created in:
    [Chetviorkin I. I. , Loukachevitch N. V. Extraction of Russian Sentiment Lexicon for Product Meta-Domain
     // In  Proceedings of COLING 2012: Technical Papers , pages 593–610]
    """
    def __init__(self, filename):
        word_to_prob = {}
        with open(filename) as f:
            for line in f:
                m = re.match('(\w+)\t([0-9.]+)', line)
                if m:
                    word = m.group(1)
                    probability = float(m.group(2))
                    word_to_prob[word] = probability

        self._word_to_prob = OrderedDict()
        self._word_to_index = {}
        for index, word in enumerate(sorted(word_to_prob.keys())):
            self._word_to_prob[word] = word_to_prob[word]
            self._word_to_index[word] = index

    def size(self):
        return len(self._word_to_prob)

    def word_pos(self, word):
        return self._word_to_index.get(word.upper())

    def words(self):
        return self._word_to_prob.keys()

    def sentiment_probability(self, word):
        return self._word_to_prob.get(word.upper(), 0.0)

    def has_sentiment(self, word):
        return self.sentiment_probability(self, word) > 0


class SimpleUnigramModel(TweetToFeaturesModel):
    model_name = "simple_unigram"

    def __init__(self, sentiment_lexicon_filename):
        self.lexicon = RussianSentimentLexicon(sentiment_lexicon_filename)

    @staticmethod
    def create_from_argv(*args):
        assert len(args) == 1
        return SimpleUnigramModel(*args)

    @staticmethod
    def load(filename):
        return SimpleUnigramModel(filename)

    def train(self, train_data):
        assert NotImplementedError("SimpleUnigramModel doesn't need any training. Simply pass sentiment lexicon file to it")

    def get_features_number(self):
        return self.lexicon.size()

    def get_features(self, tweet):
        features = [0] * self.get_features_number()
        for word in tweet.get_words():
            idx = self.lexicon.word_pos(word)
            if idx is not None:
                features[idx] = 1

        return features
