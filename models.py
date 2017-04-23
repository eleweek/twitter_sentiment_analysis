import sys
import logging
import random
import re
import os
import dill
import inspect

from collections import OrderedDict

import numpy as np

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import fasttext

import keras
import keras.preprocessing
import keras.preprocessing.text
import keras.preprocessing.sequence
from keras.utils.generic_utils import Progbar

from datasets import TweetSentiment, Tweet

_thismodule = sys.modules[__name__]

# sys.path.append("cloned_dependencies/generating-reviews-discovering-sentiment")
# import encoder as unsupervised_sentiment_neuron_encoder
# sys.path.pop()


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def find_model_class_by_name(model_class_name):
    model_class = None
    for name, obj in inspect.getmembers(_thismodule):
        if inspect.isclass(obj) and name == model_class_name or getattr(obj, "model_name", None) == model_class_name:
            model_class = obj

    if model_class is None:
        raise Exception("Unknown model name {}".format(model_class_name))

    return model_class


class TweetToFeaturesModel(object):
    @staticmethod
    def _check_features_range(features, l=-2.5, r=2.5):
        if not all(l <= f <= r for f in features):
            logging.warning("Feature outside of {} .. {} range: {}".format(l, r, features))

    def get_features_shape(self):
        raise NotImplementedError()

    def batch_get_features(self, tweets, verbose=True):
        progress = Progbar(target=len(tweets))

        X = np.zeros((len(tweets),) + self.get_features_shape(), dtype=np.float32)
        for i, tweet in enumerate(tweets):
            X[i] = self.get_features(tweets[i])
            progress.update(i)

        return X


class FasttextEmbedding(TweetToFeaturesModel):
    """
    Class for Facebook's fasttext models.
    Pre-trained models can be found here:
    https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    """
    @classmethod
    def load(cls, file_name):
        instance = cls()
        instance.model = fasttext.load_model(file_name)

        return instance

    def train(self, train_data):
        assert NotImplementedError("Fasttext currently supports only pre-trained models. You can find them here: "
                                   "https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md")

    def get_features_number(self):
        return 300  # since only pre-trained models are supported


class FasttextDocumentEmbedding(FasttextEmbedding):
    model_name = "fasttext_document_embedding"

    def get_features(self, tweet):
        features = self.model[tweet.get_text()]
        self._check_features_range(features, -100, 100)
        return features

    def get_features_shape(self):
        return (self.get_features_number(), )


class FasttextWordEmbedding(FasttextEmbedding):
    model_name = "fasttext_word_embedding"

    def __init__(self, max_words=25):
        self.max_words = max_words

    def get_features(self, tweet):
        features = np.zeros((self.get_max_words(), self.get_features_number()), dtype=np.float32)

        for i, word in enumerate(tweet.get_words()[:self.get_max_words()]):
            features[i] = self.model[word]

        return features

    def get_max_words(self):
        return self.max_words

    def get_features_shape(self):
        return (self.max_words, self.get_features_number())


class TweetSentimentModel(object):
    """
    Full model that accepts a Tweet instance and returns a sentiment
    """

    NEUTRAL_THRESHOLD = 0.33

    @staticmethod
    def _make_dirs_for_files(file_prefix):
        dirname = os.path.dirname(file_prefix)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def save(self, file_prefix):
        self._make_dirs_for_files(file_prefix)
        with open(file_prefix, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, file_prefix):
        with open(file_prefix) as f:
            instance = dill.load(f)
            assert isinstance(instance, cls)
            return instance

    def predict_sentiment_real(self, tweet):
        raise NotImplemented()

    def predict_sentiment_enum(self, tweet, without_neutral=True):
        real_value_of_sentiment = self.predict_sentiment_real(tweet)
        return TweetSentiment.from_real_value(real_value_of_sentiment, without_neutral=without_neutral)

    def is_positive(self, tweet):
        return self.get_sentiment(tweet) > self.NEUTRAL_THRESHOLD

    def is_negative(self, tweet):
        return self.get_sentiment(tweet) < -self.NEUTRAL_THRESHOLD

    def is_neutral(self, tweet):
        return not self.is_positive() and not self.is_negative()

    def train(self, tweets):
        raise NotImplemented()

    def test(self, tweets, without_neutral=True):
        """
        Tests model on a corpus of tweets.
            - without_neutral parameter removes neutral tweets from testing
            - returns accuracy
        """
        correct = 0
        total = 0
        for tweet in tweets:
            assert tweet.polarity is not None
            if tweet.is_neutral() and without_neutral:
                continue

            if tweet.polarity == self.predict_sentiment_enum(tweet, without_neutral):
                correct += 1

            total += 1

        print("correct = ", correct, "total = ", total)
        return correct / total


class FeaturesToSentimentModel(TweetSentimentModel):
    def __init__(self, tweet_to_features, features_to_sentiment=None):
        self.tweet_to_features = tweet_to_features
        self.features_to_sentiment = features_to_sentiment

    def save(self, file_prefix):
        self._make_dirs_for_files(file_prefix)
        self.save_features_to_sentiment(file_prefix)

    @classmethod
    def load(cls, loaded_tweet_to_features_model, file_prefix):
        instance = cls(loaded_tweet_to_features_model)
        instance.load_features_to_sentiment(file_prefix)

        return instance


class KerasFeaturesToSentimentModel(FeaturesToSentimentModel):
    def __init__(self, tweet_to_features, keras_model=None, batch_size=256, num_epochs=1):
        FeaturesToSentimentModel.__init__(self, tweet_to_features, keras_model)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def save_features_to_sentiment(self, file_prefix):
        self.features_to_sentiment.save(file_prefix + ".keras.h5")

    def load_features_to_sentiment(self, file_prefix):
        self.features_to_sentiment = keras.models.load_model(file_prefix + ".keras.h5")

    def predict_sentiment_real(self, tweets):
        if isinstance(tweets, Tweet):
            tweets = [tweets]
            single_instance = True
        else:
            single_instance = False

        X = self.tweet_to_features.batch_get_features(tweets)
        ys = self.features_to_sentiment.predict(X)

        if single_instance:
            return 2 * ys[0][0] - 1
        else:
            return [2 * y[0] - 1 for y in ys]

    def _features_generator(self, tweets):
        for i, chunk in enumerate(chunks(tweets, self.batch_size)):
            X = self.tweet_to_features.batch_get_features(chunk)
            y = [int(tweet.is_positive()) for tweet in chunk]
            yield X, y

    def train(self, tweets):
        self.features_to_sentiment.fit_generator(
            self._features_generator(tweets),
            steps_per_epoch=(len(tweets) // self.batch_size),
            epochs=self.num_epochs,
            verbose=1
        )


class KerasTweetSentimentModel(TweetSentimentModel):
    def __init__(self, max_words=200000, max_tweet_length=25, embedding_vector_length=300, num_epochs=1, batch_size=128, model=None):
        TweetSentimentModel.__init__(self)

        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)
        self.max_words = max_words
        self.max_tweet_length = max_tweet_length
        self.embedding_vector_length = embedding_vector_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.model = model

    def save(self, file_prefix):
        self._make_dirs_for_files(file_prefix)
        tokenizer_filename = file_prefix + ".tokenizer"
        model_params_filename = file_prefix + ".params"
        keras_model_file_name = file_prefix + ".h5"

        with open(tokenizer_filename, "wb") as tf:
            dill.dump(self.tokenizer, tf)

        with open(model_params_filename, "wb") as mpf:
            dill.dump([self.max_words, self.max_tweet_length, self.embedding_vector_length, self.num_epochs, self.batch_size], mpf)

        self.model.save(keras_model_file_name)

    @classmethod
    def load(cls, file_prefix):
        tokenizer_filename = file_prefix + ".tokenizer"
        model_params_filename = file_prefix + ".params"
        keras_model_file_name = file_prefix + ".h5"

        instance = cls()

        with open(tokenizer_filename, "rb") as tf:
            instance.tokenizer = dill.load(tf)

        with open(model_params_filename, "rb") as mpf:
            instance.max_words, instance.max_tweet_length, instance.embedding_vector_length, instance.num_epochs, instance.batch_size = dill.load(mpf)

        instance.model = keras.models.load_model(keras_model_file_name)

        return instance

    def _train_tokenizer(self, tweets):
        self.tokenizer.fit_on_texts(t.get_text() for t in tweets)

    def _tweets_to_xy_tensors(self, tweets):
        texts, y = zip(*[
            (t.get_text(), int(t.is_positive())) for t in tweets
        ])
        X = self.tokenizer.texts_to_sequences(texts)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_tweet_length)

        return X, list(y)

    def _tweets_to_x_tensor(self, tweets):
        X = self.tokenizer.texts_to_sequences(t.get_text() for t in tweets)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_tweet_length)

        return X

    def train(self, tweets, num_epochs=None, batch_size=None):
        num_epochs = num_epochs or self.num_epochs
        batch_size = batch_size or self.batch_size

        self._train_tokenizer(tweets)
        X_train, y_train = self._tweets_to_xy_tensors(tweets)
        self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size)

    def predict_sentiment_real(self, tweets):
        if isinstance(tweets, Tweet):
            Xs = self._tweets_to_x_tensor([tweets])
            ys = self.model.predict(Xs)
            return 2 * ys[0][0] - 1
        else:
            Xs = self._tweets_to_x_tensor(tweets)
            ys = self.model.predict(Xs)
            return [2 * y[0] - 1 for y in ys]


class KerasCNNModel(KerasTweetSentimentModel):
    def __init__(self, conv_layer_sizes=[128, 64, 32], dense_layer_size=180, **kwargs):
        KerasTweetSentimentModel.__init__(self, **kwargs)

        model = keras.models.Sequential()
        model.add(keras.layers.embeddings.Embedding(self.max_words, self.embedding_vector_length, input_length=self.max_tweet_length))

        for size in conv_layer_sizes:
            model.add(keras.layers.Convolution1D(size, 3, padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(dense_layer_size, activation='sigmoid'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model


class KerasLSTMModel(KerasTweetSentimentModel):
    def __init__(self, lstm_layer_sizes=[128, 32], dropout=0.2, **kwargs):
        KerasTweetSentimentModel.__init__(self, **kwargs)
        model = keras.models.Sequential()
        model.add(keras.layers.embeddings.Embedding(self.max_words, self.embedding_vector_length, input_length=self.max_tweet_length))

        for size in lstm_layer_sizes[:-1]:
            model.add(keras.layers.LSTM(size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

        model.add(keras.layers.LSTM(lstm_layer_sizes[-1], dropout=0.2, recurrent_dropout=0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


class KerasGRUModel(KerasTweetSentimentModel):
    def __init__(self, gru_layer_sizes=[128, 32], **kwargs):
        KerasTweetSentimentModel.__init__(self, **kwargs)
        model = keras.models.Sequential()
        model.add(keras.layers.embeddings.Embedding(self.max_words, self.embedding_vector_length, input_length=self.max_tweet_length))

        for size in gru_layer_sizes[:-1]:
            model.add(keras.layers.GRU(size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

        model.add(keras.layers.GRU(gru_layer_sizes[-1], dropout=0.2, recurrent_dropout=0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


class UnsupervisedSentimentNeuronEncoder(TweetToFeaturesModel):
    """
    Class representing model from OpenAI's "Learning to Generate Reviews and Discovering Sentiment":
    https://blog.openai.com/unsupervised-sentiment-neuron/

    Code with pretrained model should be put to cloned_dependencies/generating-reviews-discovering-sentiment
    """
    model_name = "unsupervised_sentiment_neuron"

    def __init__(self):
        self.model = unsupervised_sentiment_neuron_encoder.Model()

    @staticmethod
    def load(file_name):
        return UnsupervisedSentimentNeuronEncoder()

    def train(self, train_data):
        assert NotImplementedError("Unsupervised Sentiment Neuron model doesn't support training, because they haven't released the necessary code (yet ?)")

    def batch_get_features(self, tweets):
        features = self.model.transform([tweet.get_text() for tweet in tweets])
        return features

    def get_features(self, tweet):
        features = self.model.transform(tweet.get_text())
        self._check_features_range(features, -100, 100)
        return features

    def get_features_number(self):
        return 4096  # number of features in pre-trained model


class Doc2VecEmbedding(TweetToFeaturesModel):
    model_name = "doc2vec_embedding"

    def save(self, filename_prefix):
        self.model.save(filename_prefix)

    @staticmethod
    def load(filename_prefix):
        new_instance = Doc2VecEmbedding(None)
        new_instance.model = Doc2Vec.load(filename_prefix)

        return new_instance

    @staticmethod
    def create_from_argv(*args):
        return Doc2VecEmbedding(*map(int, args))

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
        return self.model[Doc2VecEmbedding._train_item_tag(self._tweet_to_index(tweet))]

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
                    word_to_prob[word.upper()] = probability

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
        return self.sentiment_probability(word) > 0

    def __contains__(self, word):
        return self.has_sentiment(word)


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
