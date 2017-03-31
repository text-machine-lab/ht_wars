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


class PopCultureCorpusLoadersTest(unittest2.TestCase):
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
