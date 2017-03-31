"""Helper functions designed to laod data from the pop culture corpus."""
import os
import config
import csv
import unittest2


def load_song_titles(filepath=os.path.join(config.POP_CULTURE_CORPUS_DIR, config.SONG_TITLES_FILE)):
    """Loads information for each popular song title in the file specified by 'filepath'.

    filepath - Absolute or relative path to pop culture song titles file

    Returns: A list of entries, one for each popular song. Each entry is formatted as
    artist, song name, year released, sales in millions, and Wikipedia sources (not useful)"""
    # Artist, Single, Released, Sales(in millions), Source
    # Right now we only care about author, song name and year
    songs = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            author = row[0]
            song_name = row[1].replace('"', '')
            year = int(row[2])
            songs.append([song_name, author, year])
    return songs


def load_movie_titles(filepath=os.path.join(config.POP_CULTURE_CORPUS_DIR, config.MOVIE_TITLES_FILE)):
    movies = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in reader:
            row_string = ' '.join(row)
            row_tokens = row_string.split()
            rank_string = ''
            for char in row_tokens[0]:
                if char.isdigit():
                    rank_string += char
            rank = int(rank_string)
            rating = float(row_tokens[-1])
            year = int(row_tokens[-2].replace('(', '').replace(')', ''))
            movie_name = ' '.join(row_tokens[1:-2])

            movies.append([movie_name, rank, year, rating])
    return movies


def load_tv_show_titles(filepath=os.path.join(config.POP_CULTURE_CORPUS_DIR, config.TV_SHOW_TITLES_FILE)):
    tv_shows = []
    with open(filepath) as f:
        for line in f:
            tokens = line.split()
            rank = int(tokens[0])
            name = ' '.join(tokens[1:])
            tv_shows.append([name, rank])
    return tv_shows


def load_book_titles(filepath=os.path.join(config.POP_CULTURE_CORPUS_DIR, config.BOOK_TITLES_FILE)):
    books = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        count = 0
        for row in reader:
            # print count
            # count += 1
            # print row
            if row[2] == 'English':
                books.append(row[:2])
    return books


def find_titles_in_tweet(tweet, titles, min_frac=0.75):
    """Returns titles contained in the tweet, each with
    a number indicating percent match.

    tweet - a single tweet, need not be lowercase
    titles - book/song/movie/tv show titles
    min_frac - function only returns titles missing min_frac * 100% of words. Default 3/4 of title missing.
    Returns: A list of entries, each entry of form [title, frac] where frac is the
    percent of title found in tweet."""
    contained_titles = []
    tweet_lower = tweet.lower()
    for title in titles:
        name = title.lower()
        num_matches = find_num_tokens_string_match(name, tweet_lower)
        percent_match = num_matches * 1.0 / len(name.split())
        if percent_match >= min_frac:
            contained_titles.append([title, percent_match])
    return contained_titles


def find_num_tokens_string_match(sub, source):
    """Calculates the percentage of substring sub can be found in string source.
    It finds if the first word in sub exists in source, and check all remaining
    words in sub to see if they exist after the first word. This function is not
    symmetric between sub and source!

    sub, source - two strings, one assumed to be a substring of the other
    Returns: the total number of matching words. Returns 0 if first word of
    sub not found in source, or if len(sub) > len(source)"""
    sub_tokens = sub.split()
    source_tokens = source.split()
    for source_token_index in range(len(source_tokens)):
        if source_tokens[source_token_index] == sub_tokens[0]:
            # We found match
            num_tokens_match = 1
            for sub_token_index in range(1, len(sub_tokens)):
                if source_token_index + sub_token_index >= len(source_tokens):
                    break
                if source_tokens[source_token_index + sub_token_index] == sub_tokens[sub_token_index]:
                    num_tokens_match += 1
            return num_tokens_match
    return 0


class PopCultureCorpusLoadersTest(unittest2.TestCase):
    def test_find_titles_in_tweet(self):
        contained_titles = find_titles_in_tweet("Harry Potter and the Big Mack blah blah",
                                                ["Harry Potter and the Sorcerer's Stone blah blah"])
        assert contained_titles[0] == ["Harry Potter and the Sorcerer's Stone blah blah", 0.75]

    def test_find_titles_in_tweet_movie_reference(self):
        books = load_book_titles()
        book_titles = [book[0] for book in books]
        tweet = "The lord of the things #DogBooks"
        contained_titles = find_titles_in_tweet(tweet, book_titles)
        assert contained_titles[0][0] == "The Lord of the Rings"

    def test_find_titles_in_tweet_double_reference(self):
        movies = load_movie_titles()
        movie_titles = [movie[0] for movie in movies]
        tweet = "it's a wonderful life now that we have run the green meter"
        contained_titles = find_titles_in_tweet(tweet, movie_titles, min_frac=.6)
        contained_title_names = [contained_title[0] for contained_title in contained_titles]
        assert "It's a Wonderful Life" in contained_title_names
        assert "The Green Mile" in contained_title_names

    def test_find_titles_in_tweet_insert_your_own_tweet_and_observe(self):
        movies = load_movie_titles()
        movie_titles = [movie[0] for movie in movies]
        your_tweet = "so many star wars"
        your_minimum_fraction_of_similarity=0.6
        contained_titles = find_titles_in_tweet(your_tweet, movie_titles, min_frac=your_minimum_fraction_of_similarity)
        print contained_titles


    def test_load_song_titles(self):
        songs = load_song_titles()
        print 'Song: %s' % str(songs[0])
        # for song in songs:
        #     print song

    def test_load_movie_titles(self):
        movies = load_movie_titles()
        print 'Movie: %s' % movies[0]
        # for movie in movies:
        #     print movie

    def test_load_tv_show_titles(self):
        tv_shows = load_tv_show_titles()
        print 'TV Show: %s' % tv_shows[0]
        # for tv_show in tv_shows:
        #     print tv_show

    def test_load_book_titles(self):
        books = load_book_titles()
        print 'Book: %s' % books[0]
        # for book in books:
        #     print book

    def test_find_num_tokens_string_match_simple(self):
        source = "I went to the grocery store today"
        sub = "I went to"
        num_matches = find_num_tokens_string_match(sub, source)
        assert num_matches == 3

    def test_find_num_tokens_string_match_with_gaps(self):
        source = "in a few weeks I will be going on a trip to Disney world"
        sub = "I can't be going to a trip in Disney"
        num_matches = find_num_tokens_string_match(sub, source)
        assert num_matches == 6

    def test_find_num_tokens_string_match_same_string(self):
        test = "Today will be a great day"
        num_matches = find_num_tokens_string_match(test, test)
        assert num_matches == len(test.split())
