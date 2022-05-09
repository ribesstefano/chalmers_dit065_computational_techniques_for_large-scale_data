from collections import defaultdict

import argparse

import numpy as np
from pybloom_live import BloomFilter

RECORDS = 1000000000
ERROR_RATE = 0.0001


def main(args):
    p = args.error_rate
    n = args.num_records

    bloom = BloomFilter(capacity=n, error_rate=p)
    duplicates = defaultdict(int)
    four_or_more_times = defaultdict(int)
    with open(args.filename) as f:
        for row in f.readlines():
            row = row.strip()
            if row in bloom:
                if row in four_or_more_times:
                    four_or_more_times[row] += 1
                elif duplicates[row] < 3:  # 3 duplicates + 1 in bloom filter = 4
                    duplicates[row] += 1
                else:
                    del duplicates[row]
                    four_or_more_times[row] = 4

            else:
                bloom.add(row)

    for d, v in four_or_more_times.items():
        print(f'{d}\t: {v}')
    return duplicates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using PySpark to obtain descriptive statistics')
    parser.add_argument('filename', default='data/Ensembl-cds-15.txt')
    parser.add_argument('--num-records', '-n',
                        default=RECORDS,
                        type=int,
                        help='How many records are to be processed')
    parser.add_argument('--error-rate', '-p',
                        type=float,
                        default=ERROR_RATE,
                        help='What false-positive rate is acceptable for the bloom filter')
    parser.add_argument('--memory', '-m',
                        type=int,
                        required=False,
                        help='If there is a memory constraint, specify here. --error-rate is ignored if this is set')

    args = parser.parse_args()
    main(args)
    # Figure out what error rate we'll get with the capacity and/or memory

    m = args.memory or -1 * (RECORDS * (np.log(ERROR_RATE))) / (np.log(2) ** 2)
    n = args.num_records or RECORDS
    error_rate = args.error_rate or ERROR_RATE

    print(m)
    print(np.log(2))
    print(m/n)
    k = np.log(2) * (m / n)
    print(ERROR_RATE, m, k)



