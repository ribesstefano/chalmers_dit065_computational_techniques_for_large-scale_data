"""
Problem 1 (5 pt): To decide the amount of memory needed for storing all words
(i.e. strings only containing the letters a-z) appearing in a large text in a
trie, compute the average length of shared prefixes between consecutive words in
a sorted list of unique words from the text. For example for the text
"Barney's barn is burning", the sorted list of words consists of barn, barneys
(removing any characters not in a-z), burning, is (note that case is ignored)
and the shared prefixes are barn, b, "" with an average of (4+1+0) / 3, in
particular: {len('barn'): 4, len('b'): 1, len(''): 0}. Implement a solution
using PySpark (or MrJob at your discretion) returning the average length of the
prefixes, the total number of unique words and the average word length. Hint: Do
not use use a sort on all words in the text (or external sort programs).  There
is text data on all machines available at /data/2022-DIT065-DAT470/gutenberg/,
e.g. /data/2022-DIT065-DAT470/gutenberg/060/06049.txt.
"""
import argparse
import os
import re
from collections import Counter

def main(sample_file):
    regex = re.compile('[^a-zA-Z ]')
    # ==========================================================================
    # Clean-up and word counter
    # ==========================================================================
    # Does using a sorted set on each line yields the same results? 
    test_set = set()
    word_counter = Counter()
    with open(sample_file, 'r') as f:
        for line in f:
            # Remove all non-aphabetical characters
            cleaned_line = regex.sub('', line.lower())
            words_in_line = cleaned_line.split(' ')
            word_counter.update(words_in_line)

            # Testing sorting on each line...
            unique_sorted_in_line = sorted(set(words_in_line))
            for w1, w2 in zip(unique_sorted_in_line, unique_sorted_in_line[1:]):
                test_set.add(os.path.commonprefix([w1, w2]))
    # ==========================================================================
    # Get unique words and their number
    # ==========================================================================
    words = list(word_counter.elements())
    print(f'num_words\t{len(words)}')
    unique_words = list(word_counter)
    print(f'num_unique_words\t{len(unique_words)}')
    # ==========================================================================
    # Average word length
    # ==========================================================================
    avg_len = sum(map(len, words)) / len(words)
    print(f'avg_word_len\t{avg_len}')
    # ==========================================================================
    # Calculate avg. prefix length
    # ==========================================================================
    # 0. Sort words
    sorted_words = sorted(unique_words)
    # 1. Make a set to avoid repetitions
    common_prefixes = set()
    # 2. Loop over each pair of consecutive words and add common prefix to set
    for w1, w2 in zip(sorted_words, sorted_words[1:]):
        common_prefixes.add(os.path.commonprefix([w1, w2]))
    common_prefixes.add(
        os.path.commonprefix([sorted_words[-2], sorted_words[-1]]))
    # 3. Once the set is built, sum the len of each prefix into an accumulator
    avg_len = sum(map(len, common_prefixes))
    # 4. Divide the accumulator by the number of found prefixes, i.e. len(set)
    avg_len /= len(common_prefixes)
    print(f'avg_prefix_len\t{avg_len}')

    # Checking and comparing the analysis on each line...
    avg_len_test = sum(map(len, test_set))
    avg_len_test /= len(test_set)
    print(f'avg_prefix_len\t{avg_len_test} ({"not " if avg_len_test != avg_len else ""}matching)')


if __name__ == '__main__':
    data_dir = '/data/2022-DIT065-DAT470/gutenberg/'
    sample_file = data_dir + '060/06049.txt'

    parser = argparse.ArgumentParser(description='Tester to obtain descriptive statistics')
    parser.add_argument('--filename',
                        default=sample_file,
                        type=str,
                        help=f'Text file to process. Default: {sample_file}')
    args = parser.parse_args()
    main(args.filename)