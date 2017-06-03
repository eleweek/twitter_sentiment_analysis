import urllib.request, urllib.parse, urllib.error,urllib.request,urllib.error,urllib.parse,json,re,datetime,sys,http.cookiejar
from .. import models
from pyquery import PyQuery
import logging

class TweetManager:
    
    def __init__(self):
        pass
        
    @staticmethod
    def getTweets(tweetCriteria, receiveBuffer = None, bufferLength = 100, user_agent=None):
        refreshCursor = ''
    
        results = []
        resultsAux = []
        cookieJar = http.cookiejar.CookieJar()

        active = True

        while active:
            json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar, user_agent)
            if len(json['items_html'].strip()) == 0:
                break

            refreshCursor = json['min_position']            
            tweets = PyQuery(json['items_html'])('div.js-stream-tweet')
            
            if len(tweets) == 0:
                break
            
            for tweetHTML in tweets:
                tweetPQ = PyQuery(tweetHTML)
                tweet = models.Tweet()
                
                usernameTweet = tweetPQ("span.username.js-action-profile-name b").text();
                txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'));
                retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
                favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
                dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
                id = tweetPQ.attr("data-tweet-id");
                permalink = tweetPQ.attr("data-permalink-path");
                user_id = int(tweetPQ("a.js-user-profile-link").attr("data-user-id"))
                
                geo = ''
                geoSpan = tweetPQ('span.Tweet-geo')
                if len(geoSpan) > 0:
                    geo = geoSpan.attr('title')
                urls = []
                for link in tweetPQ("a"):
                    try:
                        urls.append((link.attrib["data-expanded-url"]))
                    except KeyError:
                        pass
                tweet.id = id
                tweet.permalink = 'https://twitter.com' + permalink
                tweet.username = usernameTweet
                
                tweet.text = txt
                tweet.date = datetime.datetime.fromtimestamp(dateSec)
                tweet.formatted_date = datetime.datetime.fromtimestamp(dateSec).strftime("%a %b %d %X +0000 %Y")
                tweet.retweets = retweets
                tweet.favorites = favorites
                tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
                tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
                tweet.geo = geo
                tweet.urls = ",".join(urls)
                tweet.author_id = user_id
                
                results.append(tweet)
                resultsAux.append(tweet)
                
                if receiveBuffer and len(resultsAux) >= bufferLength:
                    receiveBuffer(resultsAux)
                    resultsAux = []
                
                if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
                    print("active = False")
                    active = False
                    break
                    
        
        if receiveBuffer and len(resultsAux) > 0:
            receiveBuffer(resultsAux)
        
        return results
    
    @staticmethod
    def getJsonReponse(tweetCriteria, refreshCursor, cookieJar, user_agent):
        url = "https://twitter.com/i/search/timeline?f=realtime&q=%s&src=typd&%smax_position=%s"
        
        urlGetData = ''
        if hasattr(tweetCriteria, 'username'):
            urlGetData += ' from:' + tweetCriteria.username
            
        if hasattr(tweetCriteria, 'since'):
            urlGetData += ' since:' + tweetCriteria.since
            
        if hasattr(tweetCriteria, 'until'):
            urlGetData += ' until:' + tweetCriteria.until
            
        if hasattr(tweetCriteria, 'querySearch'):
            urlGetData += ' ' + tweetCriteria.querySearch
            
        urlLang = 'l=' + tweetCriteria.lang + '&'
        # print(url)
        url = url % (urllib.parse.quote(urlGetData), urlLang, refreshCursor)

        if tweetCriteria.lang == "ru":
            accept_language = "ru,ru-RU;q=0.7,ru;q=0.3"
        elif tweetCriteria.lang == "en":
            accept_language = "en,en-US;q=0.7,en;q=0.3"
        else:
            assert "Unsupported language: ".format(tweetCriteria.lang)

        if not user_agent:
            user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.24 (KHTML, like Gecko) Ubuntu/10.04 Chromium/11.0.696.0 Chrome/11.0.696.0 Safari/534.24"

        headers = [
            ('Host', "twitter.com"),
            ('User-Agent', user_agent),
            ('Accept', "application/json, text/javascript, */*; q=0.01"),
            ('Accept-Language', accept_language),
            ('X-Requested-With', "XMLHttpRequest"),
            ('Referer', url),
            ('Connection', "keep-alive")
        ]

        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookieJar))
        opener.addheaders = headers

        try:
            response = opener.open(url, timeout=120)
            jsonResponse = response.read()
        except:
            #print("Twitter weird response. Try to see on browser: ", url)
            logging.warning("Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.parse.quote(urlGetData))
            raise
        
        dataJson = json.loads(jsonResponse.decode())
        
        return dataJson     
