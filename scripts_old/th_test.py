#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import sys
sys.path.append('/home/ppotash/semeval15/Biscuit/TaskB/code')
from predict import TwitterHawk

tweets = [l.strip().split('\t')[:2] for l in open('../data/cleaned_tweets/Add_A_Woman_Improve_A_Movie').readlines()]

th = TwitterHawk('/home/ppotash/semeval15/Biscuit/TaskB/models/trained.model')

print(th.predict(tweets))
