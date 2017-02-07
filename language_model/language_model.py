"""David Donahue 2017. n-gram language model created from Twitter data. Planned to be used
to calculate an expected GloVe embedding for the next word in a sequence."""
import unittest2


class LanguageModel:
    def __init__(self, N):
        """Creates an empty language model. Model
        does not make valid predictions. It must
        be initialized with a corpus of text or
        from file."""
        print 'Created language model'
        self.n_gram_to_word_count = {}
        self.N = N

    def initialize_model_from_text(self, lines_of_text):
        """Constructs an internal dictionary from the text, such
        that all sequences of n words or fewer are kept as keys.
        For each sequence of words, the value in the dictionary
        is a list of words and their occurrence counts.

        N - size of gram to use as context to target word"""
        print 'Created model from lines of text'
        assert isinstance(lines_of_text, list)
        assert len(lines_of_text) > 0
        # Run through all lines of text
        for line_index in range(len(lines_of_text)):
            line = lines_of_text[line_index].split(' ')
            for word_index in range(len(line)):
                word, context = self.extract_word_and_context(line, word_index)
                if context not in self.n_gram_to_word_count.keys():
                    self.n_gram_to_word_count[context] = {}
                if word not in self.n_gram_to_word_count[context].keys():
                    self.n_gram_to_word_count[context][word] = 0

                self.n_gram_to_word_count[context][word] += 1

    def extract_word_and_context(self, line, word_index):
        assert word_index >= 0
        assert word_index < len(line)
        word = line[word_index]
        if word_index > self.N:
            context = line[word_index - self.N:word_index]
        else:
            context = line[:word_index]
        # Join tokens into a string
        context = ' '.join(context)
        return word.lower(), context.lower()

    def calculate_expected_next_word(self, context):
        context = context.lower()
        context_split = context.split()
        print context, context_split
        # Only include last N tokens in context
        if len(context_split) > self.N:
            context_split = context_split[len(context_split) - self.N:]
        context = ' '.join(context_split)
        if context in self.n_gram_to_word_count:
            return self.n_gram_to_word_count[context]
        else:
            return None


class LanguageModelTest(unittest2.TestCase):
    def setUp(self):
        self.lm = LanguageModel(3)
        self.lines_of_text = ["I went to the park",
                         "I went to the museum",
                         "I went to the dentist",
                         "She went to the park",
                         "We jumped across the park",
                         "I took a bus",
                         "We took a train"]

    def test_language_model_creation(self):
        print 'Hope you can see this'
        lm = LanguageModel(3)
        self.assertTrue(len(lm.n_gram_to_word_count) == 0)

    def test_extract_word_and_context(self):
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
            num_tokens = len(line.split())
            total_words += num_tokens
        # Find number of target words
        total_expected_words = 0
        for context in self.lm.n_gram_to_word_count:
            for word in self.lm.n_gram_to_word_count[context]:
                total_expected_words += self.lm.n_gram_to_word_count[context][word]
        # Every word in text should be used as a target word
        self.assertEqual(total_expected_words, total_words)

    def test_calculate_expected_next_word(self):
        self.lm.initialize_model_from_text(self.lines_of_text)
        word_counts_dict = self.lm.calculate_expected_next_word('WENT TO THE')
        print word_counts_dict
        self.assertTrue(word_counts_dict is not None)
        self.assertTrue('park' in word_counts_dict.keys())
        self.assertTrue('museum' in word_counts_dict.keys())
        self.assertTrue('dentist' in word_counts_dict.keys())

        word_counts_dict2 = self.lm.calculate_expected_next_word('I went To THE  ')
        print word_counts_dict2
        self.assertTrue(word_counts_dict2 is not None)
        self.assertTrue('park' in word_counts_dict2.keys())
        self.assertTrue('museum' in word_counts_dict2.keys())
        self.assertTrue('dentist' in word_counts_dict2.keys())


    def tearDown(self):
        pass
