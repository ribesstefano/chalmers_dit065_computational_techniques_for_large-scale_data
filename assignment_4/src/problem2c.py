import argparse
import datetime
import math
from operator import add

import findspark
findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark import SparkConf

BINS = 10

def main(path, context, cores):

    conf = SparkConf()
    conf.setMaster("local[16]")
    conf.setAppName("your-spark-app")
    conf.set("spark.local.dir", "/data/tmp/")

    sc = SparkContext.getOrCreate(conf)



    dist_file = sc.textFile(path)
    values = dist_file.map(lambda l: l.split('\t')) \
        .map(lambda t: float(t[2]))\

    low = values.min()
    high = values.max()

    count = values.count()
    sum_ = values.sum()
    mean = sum_ / count

    variance_sum = values\
        .map(lambda x: (x - mean) ** 2) \
        .reduce(add)

    std_dev = math.sqrt(variance_sum / count)

    bin_size = (high - low) / BINS
    bins = values \
        .map(lambda v: ((v - low) // bin_size, 1)) \
        .reduceByKey(add)\
        .sortByKey()

    # Since we know the amount in each bin, we can figure out which bin the
    # median is in, which means smaller sort
    bins_list = bins.collect()
    i = 0
    bins_sum = 0
    for i, val in bins_list:
        tmp = bins_sum + val
        if tmp < count / 2:
            i += 1
            bins_sum = tmp
        else:
            break

    median_bin_low = low + (bin_size * i)
    median_bin_high = median_bin_low + bin_size

    # TODO (Huw): Can we filter the collect to be a smaller list?
    median_bin_values = values \
        .filter(lambda x: x > median_bin_low and x < median_bin_high) \
        .sortBy(lambda x: x) \
        .collect()

    median = median_bin_values[(count // 2) - bins_sum]

    output = f"The following statistics were obtained:\n" \
             f"Mean: {mean}\n" \
             f"Std Dev: {std_dev}\n" \
             f"Min: {low}\n" \
             f"Max: {high}\n" \
             f"Median: {median}\n" \
             f"The distribution of the bins was:\n"

    output += '\n'.join([f'Bin {int(i)}: {val}' for i, val in bins_list])
    print(output)


if __name__ == '__main__':

    start = datetime.datetime.now()
    # TODO (Huw): Add in argparse for datafile path and cores
    path = 'data/data-assignment-3-10M.dat'

    print((datetime.datetime.now() - start).seconds)

    parser = argparse.ArgumentParser(description='Using PySpark to obtain descriptive statistics')
    parser.add_argument('filename')
    parser.add_argument('--num-cores',
                        default='4',
                        type=int,
                        help='How many cores should be used for application')
    parser.add_argument('--runner', '-r',
                        default='local',
                        type=str,
                        help='Text to include in instantiating the SparkContext')

    args = parser.parse_args()
    main(args.filename, args.runner, args.num_cores)