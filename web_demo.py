from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from flask_wtf import Form
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

import langdetect

import twitter_utils
from datasets import Tweet
from models import KerasLSTMModel


def load_ru_model():
    return KerasLSTMModel.load("trained_models/ru_lstm_full_01")


def load_en_model():
    return KerasLSTMModel.load("trained_models/en_lstm_full_05")


twitter_api = twitter_utils.create_api()
app = Flask(__name__)
app.config['SECRET_KEY'] = "DUMMY_KEY"
Bootstrap(app)

ru_model = load_ru_model()
en_model = load_en_model()


def get_language_and_sentiment_values(text):
    lang = langdetect.detect(text)
    # if lang not in ["en", "ru"]:
    #    raise Exception("Unsupport language: {}".format(lang))

    wrapped_text = Tweet(text)
    if lang == "en":
        model = en_model
    else:
        model = ru_model

    sentiment_real = model.predict_sentiment_real(wrapped_text)
    sentiment_enum = model.predict_sentiment_enum(wrapped_text, without_neutral=False)

    return lang, sentiment_real, sentiment_enum


class TwitterSearchForm(Form):
    query = StringField('query', validators=[DataRequired()])
    Search_button = SubmitField('Search')


class UserInputForm(Form):
    text = TextAreaField('text', validators=[DataRequired()])
    get_button = SubmitField('Get sentiment')


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/search', methods=["GET", "POST"])
def search():
    form = TwitterSearchForm()
    found_tweets = None
    if form.validate_on_submit():
        found_tweets = []
        statuses = twitter_api.GetSearch(term=form.query.data, count=50)
        for status in statuses:
            text = status.text
            lang_sentiments = get_language_and_sentiment_values(text)
            found_tweets.append((text, ) + lang_sentiments)

    return render_template("search.html", form=form, found_tweets=found_tweets)


@app.route('/user_input', methods=["GET", "POST"])
def user_input():
    form = UserInputForm()
    text = lang = sentiment_real = sentiment_enum = None
    if form.validate_on_submit():
        text = form.text.data
        lang, sentiment_real, sentiment_enum = get_language_and_sentiment_values(text)

    return render_template("user_input.html", form=form, text=text, lang=lang, sentiment_real=sentiment_real, sentiment_enum=sentiment_enum)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=25000, debug=True, use_reloader=False)
