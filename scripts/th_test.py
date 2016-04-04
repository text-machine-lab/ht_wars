import sys
sys.path.append('/home/ppotash/semeval15/Biscuit/TaskB/code')
from predict import TwitterHawk

tweets = [l.strip('\n').split('\t')[:2] for l in open('../data/Add_A_Woman_Improve_A_Movie').readlines()]

th = TwitterHawk('/home/ppotash/semeval15/Biscuit/TaskB/models/trained.model')

print th.predict(tweets)
