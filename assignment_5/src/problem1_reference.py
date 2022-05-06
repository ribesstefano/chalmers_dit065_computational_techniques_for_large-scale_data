"""
Problem 1 (5 pt):

To decide the amount of memory needed for storing all words (i.e. strings only
containing the letters a-z) appearing in a large text in a trie, compute the
average length of shared prefixes between consecutive words in a sorted list of
unique words from the text.

For example for the text "Barney's barn is burning", the sorted list of words
consists of barn, barneys (removing any characters not in a-z), burning, is
(note that case is ignored) and the shared prefixes are barn, b, "" with an
average of (4+1+0) / 3, in particular: {len('barn'): 4, len('b'): 1, len(''):
0}.

Implement a solution using PySpark (or MrJob at your discretion) returning the
average length of the prefixes, the total number of unique words and the average
word length.

Hint: Do not use use a sort on all words in the text (or external sort
programs).  There is text data on all machines available at
/data/2022-DIT065-DAT470/gutenberg/, e.g.
/data/2022-DIT065-DAT470/gutenberg/060/06049.txt.

He has been advising to divide the task, and sort it in pieces rather than
sorting them at once.
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
    max_lines = -10
    i = 0
    test_set = set()
    word_counter = Counter()
    prefix_counter = Counter()
    with open(sample_file, 'r') as f:
        for line in f:
            # Remove all non-aphabetical characters
            cleaned_line = regex.sub('', line.lower())
            words_in_line = cleaned_line.split(' ')
            word_counter.update(words_in_line)

            # # Testing sorting on each line...
            # unique_sorted_in_line = sorted(set(words_in_line))
            # for w1, w2 in zip(unique_sorted_in_line, unique_sorted_in_line[1:]):
            #     prefix = os.path.commonprefix([w1, w2])
            #     test_set.add(prefix)
            #     prefix_counter.update(prefix)
            # # tmp = test_set.copy()
            # # for w in unique_sorted_in_line:
            # #     for prefix in tmp:
            # #         test_set.add(os.path.commonprefix([w, prefix]))
            i += 1
            if i == max_lines:
                break
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
    avg_word_len = sum(map(len, words)) / len(words)
    print(f'avg_word_len\t{avg_word_len}')
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
    # print(common_prefixes)

    # Checking and comparing the analysis on each line...
    avg_len_test = sum(map(len, test_set))
    avg_len_test /= len(test_set)
    matching = 'not matching' if avg_len_test != avg_len else 'matching'
    print(f'avg_prefix_len\t{avg_len_test} ({matching})')

    # ==========================================================================
    # Huw's method
    # ==========================================================================
    # 0. Extract all possible sub-strings from the unique words
    prefix_counter = Counter()
    for word in unique_words:
        prefix = ''
        for letter in word:
            prefix += letter
            prefix_counter.update([prefix])
    # 1. Eliminate substrings (i.e. prefix candidates) with count less than 2
    #    (we need to analyze the SHARED prefixes afterall)
    prefix_counter = Counter({k: c for k, c in prefix_counter.items() if c > 1})
    # 2. Remove all shared/common substrings (i.e. prefix candidates) with equal
    #    count
    prefixes = list(prefix_counter)
    for prefix, _ in prefix_counter.items():
        for i in range(len(prefix)):
            if prefix[:i] in prefix_counter and prefix_counter[prefix] == prefix_counter[prefix[:i]] and prefix[:i] in prefixes:
                prefixes.remove(prefix[:i])
    # 3. Get average length of the remaining prefixes (i.e. the prefixes are
    #    what's left from the previous step)
    avg_len_test = sum(map(len, prefixes))
    avg_len_test /= len(prefixes) + 1 # NOTE: Account for the '' shared prefix
    matching = 'not matching' if avg_len_test != avg_len else 'matching'
    print(f'avg_prefix_len\t{avg_len_test} ({matching}) [Huw"s method]')
    return avg_len_test, len(unique_words), avg_word_len


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