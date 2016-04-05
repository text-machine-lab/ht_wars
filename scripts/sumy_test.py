# -*- coding: utf8 -*-
import sys
sys.path.append('/home/ppotash/_From Bell, Eric/ark-twokenize-py/')
from twokenize import tokenizeRawTweetText
#from __future__ import absolute_import
#from __future__ import division, print_function, unicode_literals

#from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.models.dom._sentence import Sentence
from sumy.models.dom._paragraph import Paragraph
from sumy.models.dom._document import ObjectDocumentModel
#SENTENCES_COUNT = 10

class TwokenizeWrapper(object):
    def to_words(self, in_text):
        return tokenizeRawTweetText(in_text)

"""
def run_lexrank( tweets_as_string , num_tweets ):
#if __name__ == "__main__":
    #url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    #parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    LANGUAGE = "english"
    parser = PlaintextParser.from_string(tweets_as_string, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    SENTENCES_COUNT = num_tweets

    #for sentence in summarizer(parser.document, SENTENCES_COUNT):
    #    print(sentence)
    return summarizer(parser.document, SENTENCES_COUNT)
"""

tweets = [l.strip('\n').split('\t')[1] for l in open('../data/Add_A_Woman_Improve_A_Movie').readlines()]
#tweets_string = 'HEADER\n\n'+'\n\n'.join(tweets)+'\n'


sens = [Sentence(t, TwokenizeWrapper()) for t in tweets]

tweet_document = ObjectDocumentModel( [Paragraph(sens)] )

LANGUAGE = "english"
stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)
SENTENCES_COUNT = len(sens)

print summarizer(tweet_document, SENTENCES_COUNT)[sens[0]]
#print SENTENCES_COUNT
#print len(summarizer(tweet_document, SENTENCES_COUNT))


#print len(tweets)#_string

#print len(run_lexrank(tweets_string, len(tweets)))
