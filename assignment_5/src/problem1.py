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

The professor has been advising to divide the task, and sort it in pieces rather
than sorting them at once.
"""
import argparse
import datetime
import math
import os
import re
from collections import Counter
import operator
from operator import add

import findspark

findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf


def main(args):
    path, context, cores = args.filename, args.runner, args.num_cores
    # ==========================================================================
    # Config and start Spark
    # ==========================================================================
    conf = SparkConf()
    conf.setMaster('local[16]')
    conf.setAppName('your-spark-app')
    conf.set('spark.local.dir', '/tmp/')
    sc = SparkContext.getOrCreate(conf)
    # ==========================================================================
    # Load file as RDD and clean lines
    # ==========================================================================
    dist_file = sc.textFile(args.filename)
    regex = re.compile('[^a-zA-Z ]')
    words = dist_file \
        .flatMap(lambda line: regex.sub('', line.lower()).split(' ')) \
        .filter(lambda word: len(word) > 0)
    words_per_letter = words.groupBy(lambda word: word[0])
    # ==========================================================================
    # Obtain unique words, grouped by: <letter, list of words>
    # ==========================================================================
    remove_duplicates = lambda letter, words: (letter, list(set(words)))
    unique_words = words_per_letter.flatMap(remove_duplicates)
    # ==========================================================================
    # Count unique words
    # ==========================================================================
    word_counts = unique_words.reduceByKey(lambda w1, w2: len(w1) + len(w2)) \
                              .reduce(operator.add)
    print(f'Number of unique words: {word_counts}')
    # ==========================================================================
    # Average word length
    # ==========================================================================
    # words_len = words.map(lambda word: len(word))
    # words_len_count = words_len.count()
    # words_len_sum = words_len.reduce(operator.add)
    # print(f'Average word length: {words_len_sum / words_len_count}')

    # Parallelize letter-wise the words length and count
    words_len = words_per_letter.reduceByKey(lambda w1, w2: len(w1) + len(w2))
    words_len_sum = words_len.reduce(operator.add)
    words_len_count = words_per_letter.countByKey().reduce(operator.add)
    print(f'Average word length: {words_len_sum / words_len_count}')
    # ==========================================================================
    # Average prefix length
    # ==========================================================================
    sort_words = lambda letter, words: (letter, sorted(words))
    sorted_words_by_letter = unique_words.flatMap(sort_words)
    def extract_prefixes(letter, sorted_words):
        '''
        Extract a list of prefixes per alphabetic letter
        
        :param      letter:         The alphabet letter (key)
        :type       letter:         str
        :param      sorted_words:   The list of sorted words
        :type       sorted_words:   list
        
        :returns:   A list of prefixes
        :rtype:     list
        '''
        prefixes = set()
        for w1, w2 in zip(sorted_words, sorted_words[1:])
            prefixes.add(os.path.commonprefix([w1, w2]))
        return (letter, list(prefixes))
    prefixes = sorted_words_by_letter.flatMap(extract_prefixes)
    prefixes_len = prefixes.reduceByKey(lambda w1, w2: len(w1) + len(w2))
    prefixes_len = prefixes_len.reduce(operator.add)
    # prefixes_len = prefixes.map(lambda prefix: len(prefix)).reduce(operator.add)

    prefixes_count = prefixes.map(lambda letter, prefixes: (letter, len(prefixes)))
    prefixes_count = prefixes_count.reduce(operator.add)
    print(f'Average prefix length: {prefixes_len / prefixes_count}')

    # # ==========================================================================
    # # Huw's Method
    # # ==========================================================================
    # def extract_prefixes(letter, unique_words):
    #     prefixes_counter = Counter()
    #     for word in unique_words:
    #         prefix = ''
    #         for letter in word:
    #             prefix += letter
    #             prefix_counter.update([prefix])
    #     return (letter, prefixes_counter)
    # prefixes_counter = unique_words.map(extract_prefixes)
    # prefixes_len = prefixes.map(lambda prefix: len(prefix)).reduce(operator.add)
    # prefixes_count = prefixes.count()
    # print(f'Average prefix length: {prefixes_len / prefixes_count}')


if __name__ == '__main__':
    data_dir = '/data/2022-DIT065-DAT470/gutenberg/'
    sample_file = data_dir + '060/06049.txt'

    parser = argparse.ArgumentParser(description='Using PySpark to obtain descriptive statistics')
    parser.add_argument('--num-cores',
                        default='4',
                        type=int,
                        help='How many cores should be used for application')
    parser.add_argument('--runner', '-r',
                        default='local',
                        type=str,
                        help='Text to include in instantiating the SparkContext')
    parser.add_argument('--filename',
                        default=sample_file,
                        type=str,
                        help=f'Text file to process. Default: {sample_file}')
    args = parser.parse_args()
    main(args)