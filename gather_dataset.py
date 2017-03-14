import logging
FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
import twitter
from urllib.parse import urlencode
import time

api = twitter.Api(consumer_key='gU4KnfHFn3jPavX1pZxJX1gvg',
                  consumer_secret='reowuNxMSdbZgOOmR6E0FL2YgEtfziMAYGIUqjRoblqhqq0dMJ',
                  access_token_key='2563401132-skTKd7X9lWjB5uwrSZCgbKzJB9zEAJAncZ3a8SS',
                  access_token_secret='2FozNghAzJGkPXdInbK2jHtwZkpy8SkzH8wICwCwXJKZw',
                  sleep_on_rate_limit=True)

print(api.VerifyCredentials())


def dump_tweets(opened_file, tweets):
    for tweet in tweets:
        tweet_id = tweet.id
        user_name = tweet.user.screen_name
        created_at_timestamp = tweet.created_at_in_seconds
        text = tweet.text.replace('\r', '').replace('\n', ' ').replace('\t', ' ')
        opened_file.write("{}\t{}\t{}\t{}\n".format(tweet_id, user_name, created_at_timestamp, text))


def dump_tweet_dict(opened_file, tweet_dict):
    tweet_id = tweet_dict['id']
    user_name = tweet_dict["user"]["screen_name"]
    created_at = tweet_dict["created_at"]
    text = tweet_dict['text'].replace('\r', '').replace('\n', ' ').replace('\t', ' ')
    opened_file.write("{}\t{}\t{}\t{}\n".format(tweet_id, user_name, created_at, text))


def get_min_id(filename):
    res = 10**18
    with open(filename, "r", buffering=1) as f:
        for line in f:
            id = int(line.split('\t')[0])
            res = min(res, id)

    return res

min_positive_id = get_min_id("my.positive.tweets.rus.txt")
min_negative_id = get_min_id("my.negative.tweets.rus.txt")
# min_negative_id = min_positive_id  # XXX: hack

print("TEST")
logging.info("min_positive_id = {}".format(min_positive_id))
logging.info("min_negative_id = {}".format(min_negative_id))


with open("my.positive.tweets.rus.txt", "a+", buffering=1) as positive_tweets:
    with open("my.negative.tweets.rus.txt", "a+", buffering=1) as negative_tweets:
        while True:
            negative_statuses = api.GetSearch(
                raw_query=urlencode(
                    {'lang': 'ru',
                     'q': ':(',
                     'result_type': 'recent',
                     'max_id': min_negative_id - 1,
                     'count': 100}
                )
            )
            logging.info("Received {} negative tweets".format(len(negative_statuses)))
            dump_tweets(negative_tweets, negative_statuses)
            min_negative_id = min([min_negative_id] + [s.id for s in negative_statuses])

            positive_statuses = api.GetSearch(
                raw_query=urlencode(
                    {'lang': 'ru',
                        'q': ':) until:2017-03-02',
                     'result_type': 'recent',
                     'max_id': min_positive_id - 1,
                     'count': 100}
                )
            )
            logging.info("Received {} positive tweets".format(len(positive_statuses)))

            dump_tweets(positive_tweets, positive_statuses)
            min_positive_id = min([min_positive_id] + [s.id for s in positive_statuses])

            time.sleep(4)
            """
            stream = api.GetStreamFilter(track=[":)", ":("], languages=['ru'])
            for tweet_dict in stream:
                text = tweet_dict['text']
                if ':)' in text:
                    dump_tweet_dict(positive_tweets, tweet_dict)
                elif ':(' in text:
                    dump_tweet_dict(negative_tweets, tweet_dict)
                else:
                    logging.warning('Tweet without emoticon! {}'.format(text))
            """
