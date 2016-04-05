import sys
sys.path.append('/home/ppotash/_From Bell, Eric/ark-twokenize-py/')
sys.path.append('/home/ppotash/semeval15/Biscuit/TaskB/code')
from predict import TwitterHawk
from twokenize import tokenizeRawTweetText
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.models.dom._sentence import Sentence
from sumy.models.dom._paragraph import Paragraph
from sumy.models.dom._document import ObjectDocumentModel
from os import listdir
import numpy as np

class TwokenizeWrapper(object):
    def to_words(self, in_text):
        return tokenizeRawTweetText(in_text)

def get_lexrank( in_tlist ):

    sens = [Sentence(t, TwokenizeWrapper()) for t in tweets]
    tweet_document = ObjectDocumentModel( [Paragraph(sens)] )
    LANGUAGE = "english"
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    SENTENCES_COUNT = len(sens)
    lex_ranked = summarizer(tweet_document, SENTENCES_COUNT)
    if len(sens) != len(lex_ranked):
        print 'lr error'
    return [lex_ranked[s] for s in sens]

th = TwitterHawk('/home/ppotash/semeval15/Biscuit/TaskB/models/trained.model')

ht_files = listdir('../data/cleaned_tweets')
label_map = {'9':1,'1':2}


for htf in ht_files:
    print htf
    tweets = []
    labels = []
    for line in open('../data/cleaned_tweets/'+htf).readlines():
        line_split = line.strip().split('\t')
        tweets.append( line_split[:2] )
        if len(line_split) == 3:
            labels.append( label_map[line_split[2]] )
        elif len(line_split) == 2:
            labels.append(0)
        else:
            print 'error', line_split

    lex_ranks = get_lexrank( [t[1] for t in tweets] )
    sents = th.predict( tweets )

