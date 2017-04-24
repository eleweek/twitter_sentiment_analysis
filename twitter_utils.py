import os
import twitter


def create_api():
    api = twitter.Api(consumer_key=os.environ['CONSUMER_KEY'],
                      consumer_secret=os.environ['CONSUMER_SECRET'],
                      access_token_key=os.environ['ACCESS_TOKEN_KEY'],
                      access_token_secret=os.environ['ACCESS_TOKEN_SECRET'],
                      sleep_on_rate_limit=True)
    return api


def download_timeline(api, screen_name, count):
    timeline = []
    present_ids = set()
    max_id = None
    while len(timeline) < count:
        statuses = api.GetUserTimeline(screen_name='realDonaldTrump', max_id=max_id, count=5000)
        if not statuses:
            break
        for status in statuses:
            if status.id not in present_ids:
                present_ids.add(status.id)
                timeline.append(status)
        max_id = min(present_ids) - 1

    return timeline[:count]
