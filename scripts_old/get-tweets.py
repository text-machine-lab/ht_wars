import os
import sys

import twitter
# from unicode import unichar

from config import *


auth = twitter.oauth.OAuth(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
 TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
twitter_api = twitter.Twitter(auth=auth)

# _api.statuses.show(_id=282031301962395648)[u'text'])

q = '@midnight #SexySnacks'
count = 100

search_results = twitter_api.search.tweets(q=q, count=count)

statuses = search_results['statuses']

i = 0
while i < 15: # for _ in range(99):
    # print _
    print "Length of statuses", len(statuses)
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError, e:
        print 'no more'
        break

    kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])

    search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']
    i += 1
# print statuses[0]
# sys.exit()
"""
for s in statuses:
    if s[u'text'][:2] != 'RT':
        print s
sys.exit()
"""
outfile = open(os.path.join(DATA_DIR, 'Sexy_Snacks'), 'a')
for s in statuses:
    for d in s[u'entities'][u'user_mentions']:
	if d[u'name'] == '@midnight':
    		text = ''
    		for l in s[u'text'].encode('ascii', 'ignore').split('\n'):
        		text += l + ' '
    		if text[:2] != 'RT':
                        text = text.replace('\t', '').replace('\n', '')
    			outfile.write(str(s[u'id']) + '\t' + text[:-1] + '\n')
		break
outfile.close()
"""
infile = '/home/peter/Desktop/anna_projects/semeval15/past-data/SemEval2014-Task9-subtaskAB-test-to-download/SemEval2014-task9-test-B-gold-NEED-TWEET-DOWNLOAD.txt'
outfile = open('labeled-data.txt', 'a')
for line in open(infile).readlines():
    lineSplit = line.strip('\n').split('\t')
    if len(lineSplit) == 3:
        try:
            tweetText = twitter_api.statuses.show(_id = int(lineSplit[0]))[u'text'].encode('ascii','ignore')
            outfile.write(lineSplit[2]+'\t'+tweetText+'\n')
        except twitter.api.TwitterHTTPError:
            pass
    else:
        outfile.write(lineSplit[2]+'\t'+lineSplit[3]+'\n')

outfile.close()
"""
