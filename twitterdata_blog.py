import tweepy
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
import imageio

# Go to https://apps.twitter.com/ and create an app.
# The consumer key and secret will be generated for you after
consumer_key = "pj63sSUxRoR3R8USM0SKLA"
consumer_secret = "aduMWwb5J4wytrsntNvMqDCALwTQKXgbugrHJmmXpRw"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Create New App" section
access_token = "427724240-CLI4CUyybcts66ZfAs1XQW4wZg1xyH9KQNYvPQuy"
access_token_secret = "riJgO5e61GgCS0u4i4cVcuP4f8kZ1rPvCFyoDOMBY0yzh"

# Interacting with twitter's API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# creating the API object
api = tweepy.API(auth, wait_on_rate_limit=True)
results = []
for tweet in tweepy.Cursor(api.search,
                           q="giletarancioni",
                           lang="it").items(4000):
    results.append(tweet)

print(type(results))
print(len(results))
# print(results[2000].text)


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


data_set.to_csv("hashtag.csv")

Htag_df = pd.DataFrame()
j = 0

for tweet in range(0, len(results)):
    hashtag = results[tweet].entities.get('hashtags')
    for i in range(0, len(hashtag)):
        Htag = hashtag[i]['text']
        Htag_df.at[j, 'Hashtag'] = Htag
        j = j+1


Mattarella_Htag_wordcloud = Htag_df.groupby('Hashtag').size()
Mattarella_Htag_wordcloud.to_csv("mattarella_Htag_wordcloud.csv")

# Join all the text from the 1000 tweets
Hashtag_Combined = " ".join(Htag_df['Hashtag'].values.astype(str))

no_mattarella = " ".join([word for word in Hashtag_Combined.split()
                          if word != 'mattarella'
                          and word != 'Mattarella'])

Tweet_mask = imageio.imread("twitter_mask.png", as_gray=True)

# Create a Word Cloud
wc = WordCloud(background_color="white", stopwords=STOPWORDS, mask=Tweet_mask)
wc.generate(no_mattarella)
plt.imshow(wc)
plt.axis("off")
plt.savefig('mattarella_Hashtag.png', dpi=300)
plt.show()
