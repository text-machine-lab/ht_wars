import os
import logging

import requests


TWITTERHAWK_ADDRESS = 'http://twitterhawk.deephawk.org'


class TwitterHawk(object):
    def __init__(self, address, separate=True):
        self.address = address

        self.address_analyze = os.path.join(self.address, 'analyze')
        if separate:
            self.address_analyze += '?separate=1'

    def analyze(self, tweets):
        """
        Analyze the sentiment of the tweets
        Input format: [ {"id": id1, "text": text1}, {"id": id2, "text": text2}, ... ]
        """

        try:
            r = requests.post(self.address_analyze, json=tweets)
            if r.status_code == 200:
                results = r.json()

                return results['result']
            else:
                logging.error('TiwtterHawk error: %s', r.status_code)
                return None
        except ValueError as ex:
            logging.error('TiwtterHawk error: %s', ex)
            return None
        except requests.exceptions.ConnectionError as ex:
            logging.error('TiwtterHawk connection error: %s', ex)
            return None
