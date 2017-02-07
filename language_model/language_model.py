"""David Donahue 2017. n-gram language model created from Twitter data. Planned to be used
to calculate an expected GloVe embedding for the next word in a sequence."""
import unittest2
import nltk


class LanguageModel:
    def __init__(self, n):
        """Creates an empty language model. Model
        does not make valid predictions. It must
        be initialized with a corpus of text or
        from file.

        n - size of gram to use as context to target word"""
        print 'Created language model'
        self.n_gram_to_word_count = {}
        self.n = n

    def initialize_model_from_text(self, lines_of_text):
        """Constructs an internal dictionary from the text, such
        that keys are n-grams and values are dictionaries of target words.
        So for a given sequence of words, the dictionary represents words
        that are likely to come next, each associated with a count representing
        how likely it is to occur.

        - lines_of_text - a list of strings, each string is a line of text to use
        for the model"""
        print 'Created model from lines of text'
        assert isinstance(lines_of_text, list)
        assert len(lines_of_text) > 0
        # Run through all lines of text
        for line_index in range(len(lines_of_text)):
            line = lines_of_text[line_index]
            line_split = nltk.word_tokenize(line)
            for word_index in range(len(line_split)):
                word, context = self.extract_word_and_context(line_split, word_index)
                if context not in self.n_gram_to_word_count.keys():
                    self.n_gram_to_word_count[context] = {}
                if word not in self.n_gram_to_word_count[context].keys():
                    self.n_gram_to_word_count[context][word] = 0

                self.n_gram_to_word_count[context][word] += 1

    def extract_word_and_context(self, line, word_index):
        """Helper function to extract a target word and a context
        from a line of text. The target word is specified by its
        word index in the line. The context is the n or fewer words
        that come before the target word. 'n' is a parameter used
        to create this LanguageModel object.

        line - a string of text of any length
        word_index - index of target word in line (after splitting by whitespace)"""
        assert word_index >= 0
        assert word_index < len(line)
        word = line[word_index]
        if word_index > self.n:
            context = line[word_index - self.n:word_index]
        else:
            context = line[:word_index]
        # Join tokens into a string
        context = ' '.join(context)
        return word.lower(), context.lower()

    def calculate_expected_next_word(self, context):
        """Given a sequence of words, predict the next word
        by returning a list of words, each word with a count
        of how many times in the reference corpus it occurred
        after that context.

        context - a dictionary where keys are words and values are counts"""
        context = context.lower()
        context_split = nltk.word_tokenize(context)
        print context, context_split
        # Only include last N tokens in context
        if len(context_split) > self.n:
            context_split = context_split[len(context_split) - self.n:]
        context_formatted = ' '.join(context_split)
        if context_formatted in self.n_gram_to_word_count:
            return self.n_gram_to_word_count[context_formatted]
        else:
            return None


class LanguageModelTest(unittest2.TestCase):
    def setUp(self):
        """Create a language model for use in later tests."""
        self.lm = LanguageModel(3)
        self.lines_of_text = ["I went to the park",
                              "I went to the museum",
                              "I went to the dentist",
                              "She went to the park",
                              "We jumped across the park",
                              "I didn't take a bus",
                              "We took a train"]

    def test_language_model_creation(self):
        """Assume an uninitialized language model is empty."""
        print 'Hope you can see this'
        lm = LanguageModel(3)
        self.assertTrue(len(lm.n_gram_to_word_count) == 0)

    def test_extract_word_and_context(self):
        """Show that extract word and context does extract
        a word and its context correctly, that the context
        is no more than n tokens, and that the word index
        is in bounds."""
        line = ['I', 'walked', 'to', 'the', 'park', 'today']
        for index in range(len(line)):
            word, context = self.lm.extract_word_and_context(line, index)
            self.assertEqual(word, line[index].lower())
            print context
            if index == 0:
                self.assertTrue(len(context) == 0)
        with self.assertRaises(AssertionError):
            self.lm.extract_word_and_context(line, len(line))
        with self.assertRaises(AssertionError):
            self.lm.extract_word_and_context(line, -1)

    def test_initialize_model_from_text(self):
        """Show that the model produces a correct internal dictionary
        for each target word in the text. Each word in the text becomes
        a target word, with previous words in front of it becoming context.
        Show that based on the example corpus in setUp(), the model predicts
        correct counts.
        """
        self.lm.initialize_model_from_text(self.lines_of_text)
        self.assertTrue(isinstance(self.lm.n_gram_to_word_count, dict))
        self.assertTrue(isinstance(self.lm.n_gram_to_word_count['i went to'], dict))
        self.assertTrue(self.lm.n_gram_to_word_count['i went to']['the'] == 3)
        self.assertTrue('park' in self.lm.n_gram_to_word_count['went to the'])
        self.assertTrue('house' not in self.lm.n_gram_to_word_count['went to the'])
        print self.lm.n_gram_to_word_count
        print len(self.lm.n_gram_to_word_count)
        # Find number of words total
        total_words = 0
        for line in self.lines_of_text:
            num_tokens = len(nltk.word_tokenize(line))
            total_words += num_tokens
        # Find number of target words
        total_expected_words = 0
        for context in self.lm.n_gram_to_word_count:
            for word in self.lm.n_gram_to_word_count[context]:
                total_expected_words += self.lm.n_gram_to_word_count[context][word]
        # Every word in text should be used as a target word
        self.assertEqual(total_expected_words, total_words)

    def test_calculate_expected_next_word(self):
        """Show that the language model can predict the next expected words
        given previous words as context. Show this context is case-insensitive,
        and subject to nltk tokenization."""
        self.lm.initialize_model_from_text(self.lines_of_text)
        word_counts_dict = self.lm.calculate_expected_next_word('WENT TO THE')
        self.assertTrue(word_counts_dict is not None)
        self.assertTrue('park' in word_counts_dict.keys())
        self.assertTrue('museum' in word_counts_dict.keys())
        self.assertTrue('dentist' in word_counts_dict.keys())

        word_counts_dict2 = self.lm.calculate_expected_next_word('I went To THE  ')
        self.assertTrue(word_counts_dict2 is not None)
        self.assertTrue('park' in word_counts_dict2.keys())
        self.assertTrue('museum' in word_counts_dict2.keys())
        self.assertTrue('dentist' in word_counts_dict2.keys())

        word_counts_dict3 = self.lm.calculate_expected_next_word("I didn't")
        self.assertTrue(word_counts_dict3 is not None)
        self.assertTrue('take' in word_counts_dict3)
