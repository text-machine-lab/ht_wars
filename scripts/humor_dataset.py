
import os
import sys
import csv
from collections import namedtuple


class HumorDataset():
    def __init__(self, data_dir):
        self.__data_dir = data_dir
        self.__hashtags = None

    def get_hashtags(self):
        if self.__hashtags is None:
            self.__hashtags = set([
                f
                for f in os.listdir(self.__data_dir)
                if os.path.isfile(os.path.join(self.__data_dir, f))
            ])

        return self.__hashtags

    def load_hashtag(self, hashtag):
        available_hashtags = self.get_hashtags()
        if hashtag not in available_hashtags:
            raise ValueError('Bad hashtag:' + hashtag)

        Tweet = namedtuple('Tweet', ['user_id', 'content', 'score'])
        result = []

        filename = os.path.join(self.__data_dir, hashtag)
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
            for row in reader:
                result.append(Tweet(int(row[0]), row[1], int(row[2]) if row[2] != '' else 0))

        return result

    def load_hashtags(self):
        available_hashtags = self.get_hashtags()

        data = {
            ht:self.load_hashtag(ht)
            for ht in available_hashtags
        }

        return data

if __name__ == '__main__':
    data_dir = '/data1/nlp-data/ht_wars/data/cleaned_tweets/'

    hd = HumorDataset(data_dir)

    hashtags = hd.get_hashtags()
    print('Hashtags:', len(hashtags))

    data = hd.load_hashtags()
    print('Data:', len(data))

    ht_without_ones = []
    for ht, tweets in data.items():
        has_one = any([t.score == 1 for t in tweets])
        if not has_one:
            ht_without_ones.append(ht)

    print('Without ones:', len(ht_without_ones))
    for ht in ht_without_ones:
        print(ht)





