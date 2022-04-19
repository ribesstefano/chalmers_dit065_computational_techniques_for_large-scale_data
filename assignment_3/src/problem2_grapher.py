
import logging
import subprocess
from timeit import timeit

import time
import multiprocessing # See https://docs.python.org/3/library/multiprocessing.html
import numpy as np
import matplotlib.pyplot as plt
from time import process_time


WORKERS = [1, 2, 4, 8]
FILE_NOS = ['1M']


def runit(cores, filename):
    cmd = f'python3 problem2.py --num-cores={cores} --runner=local data/data-assignment-3-{filename}.dat'
    p = subprocess.run(cmd.split(' '), shell=False)


if __name__ == '__main__':
    results = {k: [] for k in FILE_NOS}
    base_perf = 1
    # for i in WORKERS:
    #     for fno in FILE_NOS:
    #         start = process_time()
    #
    #         time_perf = timeit(stmt=f'runit({i}, "{fno}")',
    #                            setup='from __main__ import runit',
    #                            number=1)
    #         print('time_perf: ', time_perf)
    #
    #         if i == 1:
    #             base_perf = time_perf
    #             res = 1
    #         else:
    #             res = base_perf / time_perf
    #
    #         results[fno].append(res)
    #         print(results)

    import numpy as np
    import matplotlib.pyplot as plt

    # data = [
    #     [1, 1],
    #     [1.2282000864859404, 1.229800314510356],
    #     [1.362055990021224, 1.361339736807973],
    #     [1.423719946773149, 1.5054990995466115],
    #     [1.4553012383502932, 1.4743593355711258],
    # ]

    data = [
        [1, 1.2282000864859404, 1.362055990021224, 1.423719946773149, 1.4553012383502932],
        [1, 1.229800314510356, 1.361339736807973, 1.5054990995466115, 1.4743593355711258],
    ]


    bar_width = 0.35
    fig = plt.subplots(figsize=(12,8))
    br1 = np.arange(len(data[0]))
    br2 = [x + 0.5 for x in br1]

    plt.bar(br1, data[0], width=bar_width, label="1M rows")
    plt.bar(br2, data[1], width=bar_width, label="10M rows")

    plt.xlabel('No. of cores running')
    plt.ylabel('Speedups')
    plt.title("Speedups of MapReduce for descriptive statistics on 1M and 10M rows")
    # plt.xticks([r + bar_width for r in data[0]], ["1M", "10M"])

    plt.legend()
    plt.show()
