import tweepy
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob, classifiers
import imageio
import argparse
import datetime

from training import training
from testing import testing

from secrets import (consumer_key, consumer_secret,
                     access_token, access_token_secret)

__version__ = '0.1-dev'

parser = argparse.ArgumentParser(description="Twitter WordCloud Analyzer "
                                 "version %s" % __version__,
                                 usage='%(prog)s -q <query_name> [options]')
parser.add_argument('-q', '--query', required=True, metavar="query_name",
                    help='target query_name (required)')
parser.add_argument('-l', '--limit', metavar='limit', type=int, default=1000,
                    help="limit the number of tweets "
                         "to retreive (default=1000)")
parser.add_argument('-L', '--lang', metavar='lang', default='it',
                    help='target lang code (default=it)')
parser.add_argument('-u', '--until', metavar='until',
                    default=datetime.date.today(),
                    help='target until date YYYY-MM-DD format. Keep in mind '
                    'that the search index has a 7-day limit. In other words, '
                    'no tweets will be found for a date older than one week. '
                    '(default=today)')
args = parser.parse_args()

query = args.query
limit = args.limit
lang = args.lang
date = args.until

date_name = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

classifier = classifiers.NaiveBayesClassifier(training)
# dt_classifier = classifiers.DecisionTreeClassifier(training)
print(classifier.accuracy(testing))
classifier.show_informative_features(3)
blob = TextBlob('mattarella non ha il nostro appoggio', classifier=classifier)
print(blob.classify())

# Interacting with twitter's API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# creating the API object
api = tweepy.API(auth, wait_on_rate_limit=True)
results = []
for tweet in tweepy.Cursor(api.search,
                           q=query,
                           until=date,
                           lang=lang).items(limit):
    results.append(tweet)


def tweets_df(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list, columns=["id"])
    data_set["text"] = [tweet.text for tweet in results]
    data_set["created_at"] = [tweet.created_at for tweet in results]
    data_set["retweet_count"] = [tweet.retweet_count for tweet in results]
    data_set["user_screen_name"] = [tweet.author.screen_name
                                    for tweet in results]
    data_set["user_followers_count"] = [tweet.author.followers_count
                                        for tweet in results]
    data_set["user_location"] = [tweet.author.location
                                 for tweet in results]
    data_set["Hashtags"] = [tweet.entities.get('hashtags')
                            for tweet in results]
    return data_set


data_set = tweets_df(results)

text = data_set["text"]

for i in range(0, len(text)):
    txt = ' '.join(word for word in text[i].split()
                   if not word.startswith('https:'))
    data_set.at[i, 'text2'] = txt

data_set.drop_duplicates('text2', inplace=True)
data_set.reset_index(drop=True, inplace=True)
data_set.drop('text', axis=1, inplace=True)
data_set.rename(columns={'text2': 'text'}, inplace=True)

text = data_set["text"]

for i in range(0, len(text)):
    textB = TextBlob(text[i])
    sentiment = textB.sentiment.polarity
    data_set.at[i, 'Sentiment'] = sentiment
    if sentiment < 0.00:
        SentimentClass = 'Negative'
        data_set.at[i, 'SentimentClass'] = SentimentClass
    elif sentiment > 0.00:
        SentimentClass = 'Positive'
        data_set.at[i, 'SentimentClass'] = SentimentClass
    else:
        SentimentClass = 'Neutral'
        data_set.at[i, 'SentimentClass'] = SentimentClass

for i in range(0, len(text)):
    textC = TextBlob(text[i], classifier=classifier)
    classify = textC.classify()
    data_set.at[i, 'Classify'] = classify
    if classify == 'neg':
        ClassifyClass = 'Negative'
        data_set.at[i, 'ClassifyClass'] = ClassifyClass
    elif classify == 'pos':
        SentimentClass = 'Positive'
        data_set.at[i, 'ClassifyClass'] = ClassifyClass
    else:
        SentimentClass = 'Neutral'
        data_set.at[i, 'ClassifyClass'] = ClassifyClass

data_set.to_csv('hashtag_' + date_name + '.csv')

Htag_df = pd.DataFrame()
j = 0

for tweet in range(0, len(results)):
    hashtag = results[tweet].entities.get('hashtags')
    for i in range(0, len(hashtag)):
        Htag = hashtag[i]['text']
        Htag_df.at[j, 'Hashtag'] = Htag
        j = j+1


Mattarella_Htag_wordcloud = Htag_df.groupby('Hashtag').size()
Mattarella_Htag_wordcloud.to_csv('query_htag_wordcloud_' + date_name + '.csv')

# Join all the text from the 1000 tweets
Hashtag_Combined = " ".join(Htag_df['Hashtag'].values.astype(str))

no_query = " ".join([word for word in Hashtag_Combined.split()
                    if word != query])

Tweet_mask = imageio.imread("twitter_mask.png", as_gray=True)

# Create a Word Cloud
wc = WordCloud(background_color="white", stopwords=STOPWORDS, mask=Tweet_mask)
wc.generate(no_query)
plt.imshow(wc)
plt.axis("off")
plt.savefig('query_hashtag_' + date_name + '.png', dpi=300)
plt.show()
