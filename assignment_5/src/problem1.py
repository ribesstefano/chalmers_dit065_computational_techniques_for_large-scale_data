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
    # ==========================================================================
    # Count unique words
    # ==========================================================================
    word_counts = words.distinct().count()
    print(f'Number of unique words: {word_counts + 1} (including empty word "")')
    # ==========================================================================
    # Average word length
    # ==========================================================================
    words_len = words.map(lambda word: len(word))
    words_len_count = words_len.count()
    words_len_sum = words_len.reduce(operator.add)
    print(f'Average word length: {words_len_sum / words_len_count}')
    # ==========================================================================
    # Average prefix length
    # ==========================================================================
    # Group words by letter and sort sublists 
    words_per_letter = words.distinct().groupBy(lambda word: word[0])
    sort_words = lambda letter_words: (letter_words[0], sorted(letter_words[1]))
    sorted_words_by_letter = words_per_letter.map(sort_words)
    # Extract prefixes from sorted sublists
    extract_prefixes = lambda x: (x[0],
        [os.path.commonprefix([w1, w2]) for w1, w2 in zip(x[1], x[1][1:])])
    remove_duplicates = lambda x: (x[0], list(set(x[1])))
    prefixes = sorted_words_by_letter.map(extract_prefixes) \
                                     .map(remove_duplicates)
    # Finally, get the requested statistics
    prefixes_len = prefixes.map(lambda x: sum([len(w) for w in x[1]])) \
                           .reduce(operator.add)
    prefixes_count = prefixes.map(lambda letter_prefix: len(letter_prefix[1])) \
                             .reduce(operator.add)
    # NOTE: We are adding +1 to consider the empty word '', which was considered
    # in the provided example and which doesn't show up in the above computation
    print(f'Average prefix length: {prefixes_len / (prefixes_count+1)}')


if __name__ == '__main__':
    data_dir = '/data/2022-DIT065-DAT470/gutenberg/'
    sample_file = data_dir + '060/06049.txt'
    program_description = 'Using PySpark to obtain descriptive statistics'

    parser = argparse.ArgumentParser(description=program_description)
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