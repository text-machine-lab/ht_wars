"""David Donahue 2016. Testing script for tree model functionality."""
from tree_model_processing import calculate_sentiment_value_of_lines
from tools import remove_hashtag_from_tweets


def main():
    print 'Testing tree model'
    # Tester functions go here.
    test_calculate_sentiment_of_lines()
    test_remove_hashtag_from_tweets()


def test_calculate_sentiment_of_lines():
    """Compare emotional tweets to confirm
    proper sentiment calculation."""
    example_tweet1 = 'I am having a bad day'
    example_tweet2 = 'I am having a good day'
    example_tweet3 = 'I am having an average day'
    tweets = [example_tweet1, example_tweet2, example_tweet3]
    tweet_sentiments = calculate_sentiment_value_of_lines(tweets)
    assert tweet_sentiments[1] > tweet_sentiments[2]
    assert tweet_sentiments[2] > tweet_sentiments[0]


def test_remove_hashtag_from_tweets():
    """Check that hashtag is removed."""
    tweets = ['What a day #Tired', 'Going #swimming in my #pool', '#beginning of tweet']
    tweets_without_hashtags = remove_hashtag_from_tweets(tweets)
    assert tweets_without_hashtags[0] == 'What a day '
    assert tweets_without_hashtags[1] == 'Going in my '
    assert tweets_without_hashtags[2] == 'of tweet'


if __name__ == '__main__':
    main()