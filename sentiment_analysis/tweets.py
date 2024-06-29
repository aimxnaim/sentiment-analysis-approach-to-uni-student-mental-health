import snscrape.modules.twitter as sntwitter 
import pandas as pd

query = "python"
tweet_texts = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    tweet_texts.append(tweet.content)

# Display the extracted tweet texts
print(tweet_texts)
