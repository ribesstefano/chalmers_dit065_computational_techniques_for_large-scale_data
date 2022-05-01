import argparse
import logging
import time
import os
import multiprocessing # See https://docs.python.org/3/library/multiprocessing.html
import subprocess
import random
from math import sqrt
from time import process_time
from timeit import timeit
from operator import itemgetter

class Point:
    def __init__(self, x=0.0, y=0.0, cluster=0):
        self.x, self.y, self.cluster = float(x), float(y), cluster

    def __repr__(self):
        return f'({self.x}, {self.y})'


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dists = []
    for c in centroids:
        dists.append(sqrt((datum.x - c.x)**2 + (datum.y - c.y)**2))
    return min(enumerate(dists), key=itemgetter(1))


def kmeansSerial(k, data, nr_iter=100, centroids_init=None):
    N = len(data)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = [0] * N
    if centroids_init is None:
        # Choose k random data points as centroids
        centroids = random.sample(data, k)
    else:
        centroids = centroids_init
    for j in range(nr_iter):
        cluster_sizes = [0] * k
        for i in range(N):
            cluster, dist = nearestCentroid(data[i], centroids)
            c[i] = cluster
            cluster_sizes[cluster] += 1
        # Recompute centroids
        centroids = [Point(0, 0)] * k
        for i in range(N):
            centroids[c[i]].x += data[i].x        
            centroids[c[i]].y += data[i].y
        for i in range(k):
            # print(cluster_sizes[i], centroids[i])
            centroids[i].x = centroids[i].x / cluster_sizes[i]
            centroids[i].y = centroids[i].y / cluster_sizes[i]
            print(cluster_sizes[i], centroids[i])
    return centroids


def run_jobs(cores, points_file, centroids_infile, centroids_outfile=None):
    cmd = f'python3 problem1.py --num-cores={cores} --runner=local '
    # cmd += f'--step-num=1 '
    cmd += f'--centroids {centroids_infile} '
    cmd += f'{points_file} '
    if centroids_outfile is not None:
        cmd += f'> {centroids_outfile} '
    subprocess.run(cmd, shell=True)


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    centroids_infile = os.path.join(curr_dir, 'centroids.dat')
    centroids_outfile = os.path.join(curr_dir, 'centroids.out')
    # Generate centroids file by randomily sample k points
    data = []
    with open(args.filename, 'r') as fd:
        for line in fd:
            x, y = line.rstrip().split('\t')
            data.append(Point(x, y))
    centroids = random.sample(data, args.k_clusters)
    with open(centroids_infile, 'w') as fc:
        for i, c in enumerate(centroids):
            fc.write(f'{c.x}\t{c.y}')
            if i != args.k_clusters - 1:
                fc.write('\n')
    print('DATAPOINT: ', args.filename)
    print('CENTROIDS: ', centroids_infile)
    print('-' * 80)
    # Run MRJobs
    run_jobs(cores=args.workers,
             points_file=args.filename,
             centroids_infile=centroids_infile,
             centroids_outfile=centroids_outfile)
    # Run serial implementation
    centroids_serial = kmeansSerial(k=args.k_clusters,
                                    data=data,
                                    centroids_init=centroids,
                                    nr_iter=1)
    # Check results
    centroids_parallel = []
    with open(centroids_outfile, 'r') as fc:
        for line in fc:
            x, y = line.split('\t')
            centroids_parallel.append(Point(x, y))
    num_errors = 0
    for c_seq, c_par in zip(centroids_serial, centroids_parallel):
        if c_seq.x != c_par.x or c_seq.y != c_par.y:
            num_errors += 1
            # print(f'expected/got: {c_seq} / {c_par}')
    print('-' * 80)
    print(f'Number of mismatches: {num_errors}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: problem1_runjobs.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes to use')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        default='100',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        default='10000',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')   
    parser.add_argument('--filename', '-f',
                        default='kmeans.dat',
                        type = str,
                        help='K-means data point filename')
    parser.add_argument('--plot', '-p',
                        type = str,
                        help='Filename to plot the final result')   
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()
    main(args)

    # # data = [
    # #     [1, 1],
    # #     [1.2282000864859404, 1.229800314510356],
    # #     [1.362055990021224, 1.361339736807973],
    # #     [1.423719946773149, 1.5054990995466115],
    # #     [1.4553012383502932, 1.4743593355711258],
    # # ]

    # data = [
    #     [1, 1.2282000864859404, 1.362055990021224, 1.423719946773149, 1.4553012383502932],
    #     [1, 1.229800314510356, 1.361339736807973, 1.5054990995466115, 1.4743593355711258],
    # ]


    # bar_width = 0.35
    # fig = plt.subplots(figsize=(12,8))
    # br1 = np.arange(len(data[0]))
    # br2 = [x + 0.5 for x in br1]

    # plt.bar(br1, data[0], width=bar_width, label="1M rows")
    # plt.bar(br2, data[1], width=bar_width, label="10M rows")

    # plt.xlabel('No. of cores running')
    # plt.ylabel('Speedups')
    # plt.title("Speedups of MapReduce for descriptive statistics on 1M and 10M rows")
    # # plt.xticks([r + bar_width for r in data[0]], ["1M", "10M"])

    # plt.legend()
    # plt.show()