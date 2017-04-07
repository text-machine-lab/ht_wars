"""David Donahue 2017. Takes in the CMU movie corpus and parses it for titles and dates of movies(pontentially other attributes).
Prints to stdout, so result can be saved to file."""

import fileinput
import sys

count = 1
for line in fileinput.input(sys.argv[1:]):
    line_tokens = line.split('\t')
    if (line.find("English") != -1 or line.find("Language") == -1) and line_tokens[4] != '':
        count += 1
        print line_tokens[2] + "\t" + line_tokens[3] + "\t" + line_tokens[4]