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

BINS = 10

def main(args):
    path, context, cores = args.filename, args.runner, args.num_cores
    # Config and start Spark
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
    # Obtain unique words, grouped by: <letter, list of words>
    # ==========================================================================
    remove_duplicates = lambda letter, words: (letter, set(words))
    unique_words = words.groupBy(lambda word: word[0]) \
                        .groupByKey(remove_duplicates)
    # ==========================================================================
    # Count unique words
    # ==========================================================================
    word_counts = unique_words.reduceByKey(lambda letter, words: len(words)) \
                              .reduce(operator.add)
    print(f'Number of unique words: {word_counts}')
    # ==========================================================================
    # Average word length
    # ==========================================================================
    words_len = words.map(lambda word: len(word))
    words_len_count = words_len.count()
    words_len_sum = words_len.reduce(operator.add)
    print(f'Average word length: {words_len_sum / words_len_count}')
    # ==========================================================================
    # Average prefix length (TODO)
    # ==========================================================================
    sort_set = lambda letter, words: (letter, sorted(words))
    sorted_words_by_letter = unique_words.groupByKey(sort_set)

    # for letter, words in sorted_words_by_letter.collect():
    #     print(letter, words.data)
        # .reduceByKey(lambda word: (word, 1)).collect()

    # unique_words_list = unique_words # .collect()


    # values = dist_file.map(lambda l: l.split('\t')) \
    #     .map(lambda t: float(t[2]))

    # low = values.min()
    # high = values.max()

    # count = values.count()
    # sum_ = values.sum()
    # mean = sum_ / count

    # variance_sum = values\
    #     .map(lambda x: (x - mean) ** 2) \
    #     .reduce(add)

    # std_dev = math.sqrt(variance_sum / count)

    # bin_size = (high - low) / BINS
    # bins = values \
    #     .map(lambda v: ((v - low) // bin_size, 1)) \
    #     .reduceByKey(add)\
    #     .sortByKey()

    # # Since we know the amount in each bin, we can figure out which bin the
    # # median is in, which means smaller sort
    # bins_list = bins.collect()

    # output = f"The following statistics were obtained:\n" \
    #          f"Mean: {mean}\n" \
    #          f"Std Dev: {std_dev}\n" \
    #          f"Min: {low}\n" \
    #          f"Max: {high}\n" \
    #          f"The distribution of the bins was:\n"

    # output += '\n'.join([f'Bin {int(i)}: {val}' for i, val in bins_list])
    # print(output)


if __name__ == '__main__':
    data_dir = './data/'
    sample_file = data_dir + '00001.txt'

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