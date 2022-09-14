import click
import numpy as np
import pickle
import pprint
import os
import re

from functools import wraps
from itertools import groupby
from operator import itemgetter
from gensim.models import Word2Vec
from scipy.spatial import distance
from utils import api


USERS = [
    'Jason',
    'mollywood',
    'brikeilarcnn',
    'DavidSacks',
    'TuckerCarlson',
    'pmarca',
    'peterthiel',
    'benshapiro',
]


def filecache(func):
    @wraps(func)
    def wrapper(users, filename, refresh=False):
        """
        If the filename already exists, load it; if not, proceed with the wrapped function.
        """
        if not refresh:
            try:
                with open(os.path.join('static', filename), 'rb') as f:
                    return pickle.load(f)
            except:
                refresh = True
            
        if refresh:
            data = list(func(users, filename, refresh))
            with open(os.path.join('static', filename), 'wb') as f:
                pickle.dump(data, f)
            return data

    return wrapper

@filecache
def get_tweets(users, filename, refresh):
    """
    Given a list of Twitter handles, return an iterable of (tweet, user)

    Requirements
    ============
    users : iterable
    filename : str
    refresh: bool
    """
    for u in users:
        results = api.user_timeline(screen_name=u)
        for r in results:
            yield r.text, r.author.screen_name


class TopicModelPipeline(object):
    def __init__(self, data, filename, word_vectors=None):
        """
        data is a list of tuples, where a tuple is (tweet, author)

        Example
        =======
        ['I am feeling good', 'jack',
         'buying twitter for 54.20 #blessed', 'elonmusk']
        """
        self.tweets = [i[0] for i in data]
        self.author = [i[1] for i in data]
        self.data = data
        self.filename = filename
        if word_vectors:
            import gensim.downloader

            self.kv = gensim.downloader.load(word_vectors)
    
    def clean_text(self, tweets):
        """
        Given a list of tweets, perform the following cleaning steps:
        - remove @ mentions
        - remove URLs
        - remove 'RT' from retweets
        - lower-case everything
        - tokenize on space, returning a list of lists

        e.g. ['I feel good',
              'I am buying Twitter']
        becomes:
            [
                ['I', 'feel', 'good],
                ['I', 'am', 'buying', 'Twitter'],
            ]
        """
        replacements = [
            (r'@(\w+)', ''),
            (r'http(.+)\s?', ''),
            (r'RT\s', ''),
        ]
        for tweet in tweets:
            for old, new in replacements:
                tweet = re.sub(old, new, tweet)
            yield tweet.lower().split(' ')
    
    def train_model(self):
        """
        Train a gensim word2vec model based on input tweets
        """
        model = Word2Vec(sentences=list(self.clean_text(self.tweets)),
                         min_count=2)
        model.save(os.path.join('static', self.filename + '.model'))
        model.wv.save(os.path.join('static', self.filename + '.kv'))
        self.kv = model.wv
    
    def vectorize_users(self):
        """
        Return a list of vectors for the list of users provided
        """
        self.user_vectors = {}
        for user, grp in groupby(self.data, itemgetter(1)):
            user_tweets = []
            for t, u in grp:
                cleaned = self.clean_text(t)
                for word in t:
                    try:
                        wv = self.kv[word]
                    except KeyError:  # word is not in keyed-vector vocabulary
                        continue
                    else:
                        user_tweets.append(wv)

            self.user_vectors[user] = np.mean(user_tweets, axis=0)
        return self.user_vectors
    
    def _calculate_user_similarity(self, user):
        for other_user, ouv in self.vectorize_users().items():
            if other_user == user:
                continue

            yield other_user, distance.euclidean(self.user_vectors[user], ouv)
    
    def find_most_similar_users(self, user):
        """
        Given an input user, find the most similar users from the input data based on Euclidean distance
        """
        users = self._calculate_user_similarity(user)
        yield sorted(users, key=itemgetter(1))

@click.command()
@click.option('--filename')
@click.option('--refresh/--no-refresh', default=False)
@click.option('--word_vectors', type=click.Choice([
    'glove-twitter-25'
]))
def main(filename, refresh, word_vectors):
    data = list(get_tweets(USERS, filename, refresh))
    pipeline = TopicModelPipeline(data, filename, word_vectors)
    if not word_vectors:
        pipeline.train_model()

    user_vectors = pipeline.vectorize_users()

    print('Closest users to benshapiro')
    pprint.pprint(list(pipeline.find_most_similar_users('benshapiro')))
    print('---')
    print('Closest users to mollywood')
    pprint.pprint(list(pipeline.find_most_similar_users('mollywood')))

if __name__ == '__main__':
    main()