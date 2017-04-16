import os
import csv
import logging
from datetime import timedelta, datetime
import argparse

from fake_useragent import UserAgent as FakeUserAgent

import got3

FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def get_min_date(filename, date_field_num=1):
    # TBH it's probably better to simply return the last date
    min_date = None
    with open(filename) as f:
        for line in f:
            date_field = line.split(';')[date_field_num]
            if min_date is None or date_field < min_date:
                min_date = date_field

    return date_from_str(min_date)


def get_last_query_and_date(filename, query_field_num=0, date_field_num=3):
    with open(filename) as f:
        for line in f:
            fields = line.split(';')
            query = fields[query_field_num]
            date = fields[date_field_num]

    # Simply return query and date from the last line
    logging.debug(
        "get_last_query_and_date(): query_field_num = {}; date_field_num = {}; query = {}; date = {};".format(
            query_field_num,
            date_field_num,
            query,
            date)
    )
    return query, date_from_str(date)


def date_from_str(date_str):
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception as e:
            logging.debug("Exception while converting str to date: {}".format(e))

    logging.warning("Failed to convert string {} to date using formats {}".format(date_str, formats))

    return None


class WriteTweetsToCSV(object):
    def __init__(self, csv_writer, include_query_and_polarity):
        self.csv_writer = csv_writer
        self.last_datetime_written = None
        self.query = None
        self.polarity = None

        self.include_query_and_polarity = include_query_and_polarity

        self.total_tweets_written_count = 0

    def __call__(self, tweets):
        logging.info("Got {} tweets: from {} to {}".format(len(tweets), tweets[-1].date, tweets[0].date))

        for t in tweets:
            row = (t.username,
                   t.date.strftime("%Y-%m-%d %H:%M"),
                   t.retweets,
                   t.favorites,
                   t.text,
                   t.geo,
                   t.mentions,
                   t.hashtags,
                   t.id,
                   t.permalink)

            if self.query or self.polarity:
                if not self.include_query_and_polarity:
                    raise Exception("self.query is present, but include_query option is not set")
                row = (self.query, self.polarity) + row
            self.csv_writer.writerow(row)
            self.total_tweets_written_count += 1

            if self.last_datetime_written is not None:
                self.last_datetime_written = min(t.date, self.last_datetime_written)
            else:
                self.last_datetime_written = t.date

    def get_last_datetime_written(self):
        return self.last_datetime_written

    def set_last_datetime_written(self, last_datetime_written):
        self.last_datetime_written = last_datetime_written

    def reset_last_datetime_written(self):
        self.last_datetime_written = None

    def set_current_query(self, query):
        self.query = query

    def set_current_polarity(self, polarity):
        self.polarity = polarity

    def get_total_tweets_written_count(self):
        return self.total_tweets_written_count


def read_query_list_from_file(query_list_file):
    queries = []
    with open(query_list_file, 'rt') as f:
        for line in f:
            queries.append(line.strip())

    return queries


def download_all_tweets_from_query(query, write_tweets_to_csv, date_since, date_until, lang, user_agent):
    tweet_criteria = got3.manager.TweetCriteria()
    tweet_criteria.lang = lang
    tweet_criteria.querySearch = query
    date_format = '%Y-%m-%d'
    tweet_criteria.since = date_since.strftime(date_format)
    tweet_criteria.until = date_until.strftime(date_format)

    last_datetime = date_until
    empty_days_count = 0
    number_of_empty_days_before_early_stop = 30

    while tweet_criteria.since < tweet_criteria.until and empty_days_count < number_of_empty_days_before_early_stop:
        logging.info("Starting new tweet download session {}..{}".format(tweet_criteria.since, tweet_criteria.until))
        try:
            logging.info("Getting tweets...")
            got3.manager.TweetManager.getTweets(tweet_criteria, write_tweets_to_csv, user_agent=user_agent)
            logging.info("Done getting tweets...")
        except SystemExit:
            raise
        except:
            logging.exception("Exception while getting tweets")
        prev_datetime = last_datetime
        last_datetime = write_tweets_to_csv.get_last_datetime_written() or date_until

        if prev_datetime != last_datetime:
            empty_days_count = 0
        else:
            empty_days_count += 1

        last_datetime -= timedelta(days=1)
        print(last_datetime)

        # sometimes it is necessary to skip empty days
        # we also update decrease last_datetime to avoid infinite loops because of long empty periods
        write_tweets_to_csv.set_last_datetime_written(last_datetime)

        tweet_criteria.until = last_datetime.strftime(date_format)
    logging.info("Successfully finished downloading requested tweets")


def main():
    parser = argparse.ArgumentParser(prog="Скачивание твитов в обход Twitter API")
    parser.add_argument("--query",
                        default=None,
                        help="Скачать твиты, соответствующие этому запросу. Опция несовместима с --query-list")
    parser.add_argument("--query-list",
                        default=None,
                        help="Файл с запросами для поиска. Запросы должны идти по одному на строку. Опция несовместима с --query")
    parser.add_argument("--query-list-mode",
                        choices=["raw", "emoticons"],
                        help="Режим скачивания списка запросов. В режиме 'raw' запросы задаются без изменения. "
                             "В режиме 'emoticons' запрос [query] превращается в два запроса [query :)] и [query :(] --"
                             " то есть каждый запрос задается два раза, один раз со смайликом :) и второй раз со смайликом :("
                        )
    parser.add_argument("--date-since",
                        required=True,
                        help="Начало временного периода, в которой должны попадать твиты. Формат: YYYY-MM-DD")
    parser.add_argument("--date-until",
                        required=True,
                        help="Окончание временного периода, в который должны попадать твиты. Формат: YYYY-MM-DD")
    parser.add_argument("--out-file",
                        required=True,
                        help="Файл, в который записывать результаты")
    parser.add_argument("--user-agent",
                        default=None,
                        help="Строка User Agent. По умолчанию случайным образом выбирается реальный User Agent"
                        )
    parser.add_argument("--lang",
                        choices=["ru", "en"],
                        default="ru",
                        help="Язык твитов. Поддерживается русский и английский"
                        )
    parser.add_argument("--dont-continue",
                        default=False,
                        help="По-умолчанию программа поддерживает режим докачивания. "
                             "Эта опция отключает данное поведение (выходной файл будет перезаписан)"
                        )

    args = parser.parse_args()
    if args.query_list is not None and args.query is not None:
        raise Exception("--query-list and --query are mutually incompatible")

    if args.query_list is None and args.query is None:
        raise Exception("Must specify exactly one of --query-list and --query")

    user_agent = args.user_agent
    if user_agent is None:
        try:
            fua = FakeUserAgent()
            user_agent = fua.random
            logging.info("Trying to download tweets with the following user-agent: {}".format(user_agent))
        except:
            logging.exception("Cannot generate fake user agent. Falling back to some default user agent")

    query = args.query
    query_list_file = args.query_list
    query_list_download_mode = args.query_list_mode
    out_file_name = args.out_file

    date_since_str = args.date_since
    date_until_str = args.date_until

    date_since = date_from_str(date_since_str)
    date_until = date_from_str(date_until_str)
    dont_continue = args.dont_continue
    lang = args.lang
    logging.info("date_since = {}; date_until = {}; lang = {}".format(date_since, date_until, lang))

    out_file_mode = "wt" if dont_continue else "at"

    assert (query is not None) != (query_list_file is not None)

    if query:
        if not dont_continue and os.path.exists(out_file_name) and os.stat(out_file_name).st_size > 0:
            logging.info("Output file exists, trying to continue gathering tweets")
            date_until = min(date_until, get_min_date(out_file_name) - timedelta(days=1))
            logging.info("Adjusting date_until = {}".format(date_until))

        with open(out_file_name, out_file_mode) as out_file:
            csv_writer = csv.writer(out_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            write_tweets_to_csv = WriteTweetsToCSV(csv_writer, include_query_and_polarity=False)
            download_all_tweets_from_query(query,
                                           write_tweets_to_csv,
                                           date_since,
                                           date_until,
                                           lang=lang,
                                           user_agent=user_agent)

    elif query_list_file:
        queries = read_query_list_from_file(query_list_file)

        if not dont_continue and os.path.exists(out_file_name) and os.stat(out_file_name).st_size > 0:
            logging.info("Output file exists, trying to continue gathering tweets")
            continue_from_query, last_date = get_last_query_and_date(out_file_name)
            continue_from_date = last_date - timedelta(days=1)
            logging.info("Should continue from query {} and {}".format(continue_from_query, continue_from_date))
        else:
            continue_from_date = None
            continue_from_query = None

        if query_list_download_mode == "raw":
            emoticons_and_polarities = [('', None)]
        else:
            emoticons_and_polarities = [(':)', 1), (':(', -1)]

        with open(out_file_name, out_file_mode) as out_file:
            csv_writer = csv.writer(out_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            write_tweets_to_csv = WriteTweetsToCSV(csv_writer, include_query_and_polarity=True)

            skip_queries = not dont_continue and continue_from_query is not None
            for query in queries:
                for emoticon, polarity in emoticons_and_polarities:
                    if emoticon:
                        actual_twitter_query = "{} {}".format(query, emoticon)
                    else:
                        actual_twitter_query = query
                    logging.debug("actual_twitter_query = {}".format(actual_twitter_query))

                    if actual_twitter_query == continue_from_query:
                        skip_queries = False
                    if skip_queries:
                        logging.debug("Skipping actual_twitter_query = {}".format(actual_twitter_query))
                        continue

                    write_tweets_to_csv.set_current_polarity(polarity)
                    write_tweets_to_csv.set_current_query(actual_twitter_query)
                    write_tweets_to_csv.reset_last_datetime_written()
                    download_all_tweets_from_query(actual_twitter_query,
                                                   write_tweets_to_csv,
                                                   date_since,
                                                   date_until if actual_twitter_query != continue_from_query else continue_from_date,
                                                   lang=lang,
                                                   user_agent=user_agent)
    else:
        assert False

    if write_tweets_to_csv.get_total_tweets_written_count() == 0:
        logging.critical("Nothing downloaded! If this is unexpected, consider overwriting user agent with --user-agent option. "
                         "Sometimes twitter bans by user agent")


if __name__ == "__main__":
    main()
