#!/usr/bin/env python
#
# File: kmeans.py
# Authors: Stefano Ribes (ribes@chalmers.se, Alexander Schliep (alexander@schlieplab.org)
#
#
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing # See https://docs.python.org/3/library/multiprocessing.html
from sklearn.datasets import make_blobs


def generateData(n, c):
    logging.info(f'Generating {n} samples in {c} classes')
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def amdhals_law(T_seq, T_par, p):
    return (T_seq + T_par) / (T_seq + T_par / p)


def kmeansSerial(k, data, nr_iter=100, p=1):
    np.random.seed(1)

    N = len(data)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info('Iteration\tVariation\tDelta Variation')
    total_variation = 0.0
    nearestCentroid_t = 0
    recomputeCentroids_t = 0
    # ==========================================================================
    # Candidate Parallel region:
    # * The centroids vector is iteratively updated, meaning that the value of
    #   centroids depends on the previous iteration. Moreover, it is used to
    #   calculate the distances, so the iterations must be executed sequentially
    # * c (i.e. the cluster indeces) is shared and updated at each iteration.
    #   NOTE: Having it defined INSIDE the loop would make no differece though.
    # ==========================================================================
    start_time_iters = time.time()
    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)), size=k, replace=False)]
    logging.debug('Initial centroids\n', centroids)
    for j in range(nr_iter):
        logging.debug('=== Iteration %d ===' % (j+1))
        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        # ======================================================================
        # Candidate Parallel region:
        # * The centroids vector is shared, but never modified, so a copy can be
        #   passed to the workers.
        # * c (i.e. the cluster indeces) is also shared, but each worker only
        #   modifies a certain pool of indeces.
        # * cluster_sizes and variation are shared and the indeces of the values
        #   to be modified can be shared across the workers too. So a mux on
        #   them might be needed.
        start_time = time.time()
        for i in range(N):
            cluster, dist = nearestCentroid(data[i], centroids)
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2
        # ======================================================================
        end_time = time.time()
        nearestCentroid_t += end_time - start_time
        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        logging.info('%3d\t\t%f\t%f' % (j, total_variation, delta_variation))

        # Recompute centroids
        centroids = np.zeros((k, 2)) # This fixes the dimension to 2
        # ======================================================================
        # Candidate Parallel region:
        # * The centroids vector is shared and updated by all workers, so there
        #   might be needed muxes on it (potentially generating too much
        #   overhead)
        start_time = time.time()
        for i in range(N):
            centroids[c[i]] += data[i]        
        # ======================================================================
        centroids = centroids / cluster_sizes.reshape(-1, 1)
        end_time = time.time()
        recomputeCentroids_t += end_time - start_time
        
        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
    stop_time_iters = time.time()
    total_time = stop_time_iters - start_time_iters
    serial_time = total_time - nearestCentroid_t - recomputeCentroids_t
    logging.info(f'Total time:                    {total_time:3.2f} [s]')
    logging.info(f'Total serial time:             {serial_time:3.2f} [s] (f_seq:{serial_time / total_time:.2f})')
    logging.info(f'Total nearestCentroid time:    {nearestCentroid_t:3.2f} [s] (f_par:{nearestCentroid_t / total_time:.2f})')
    logging.info(f'Total recomputeCentroids time: {recomputeCentroids_t:3.2f} [s] (f_par:{recomputeCentroids_t / total_time:.2f})')
    logging.info(f'Avg. nearestCentroid time:     {nearestCentroid_t / nr_iter:3.2f} [s]')
    logging.info(f'Avg. recomputeCentroids time:  {recomputeCentroids_t / nr_iter:3.2f} [s]')

    logging.info('=' * 80)
    T_seq = total_time - nearestCentroid_t
    T_par = nearestCentroid_t
    logging.info(f'Amdhal Sp(nearestCentroid, {p}) = {amdhals_law(T_seq, T_par, p):.3f}')

    T_seq = total_time - recomputeCentroids_t
    T_par = recomputeCentroids_t
    logging.info(f'Amdhal Sp(recomputeCentroids, {p}) = {amdhals_law(T_seq, T_par, p):.3f}')
    logging.info('=' * 80)
    return total_variation, c


def computeDistances(n_low, n_high, k, data, centroids):
    variation = np.zeros(k)
    cluster_sizes = np.zeros(k, dtype=int)
    for i in range(n_low, n_high):
        cluster, dist = nearestCentroid(data[i], centroids)
        c[i] = cluster # NOTE: The c array is shared across workers
        cluster_sizes[cluster] += 1
        variation[cluster] += dist**2
    return variation, cluster_sizes


def recomputeCentroids(n_low, n_high, k, data):
    centroids = np.zeros((k, 2)) # This fixes the dimension to 2
    for i in range(n_low, n_high):
        centroids[c[i]] += data[i]
    return centroids


def split(a, n):
    k, m = divmod(len(a), n)
    return ((i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n))


def kmeans(k, data, nr_iter=100, num_workers=1):
    np.random.seed(1)

    N = len(data)
    # Choose k random data points as centroids
    rand_idx = np.random.choice(np.array(range(N)), size=k, replace=False)
    centroids = data[rand_idx]
    logging.debug('Initial centroids\n', centroids)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    # NOTE: The cluster index variable, c, is made global and sharable across
    #       the workers.
    # NOTE: The multiprocessing pool is created OUTSIDE the main loop.
    global c
    c = multiprocessing.Array('i', [0] * N)
    p = multiprocessing.Pool(num_workers)

    logging.info('Iteration\tVariation\tDelta Variation')
    total_variation = 0.0
    for j in range(nr_iter):
        logging.debug('=== Iteration %d ===' % (j+1))
        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)

        slices = list(split(list(range(N)), num_workers))
        worker_args = [(lo, hi, k, data, centroids) for (lo, hi) in slices]
        ret_list = p.starmap(computeDistances, worker_args)
        for var, c_sizes in ret_list:
            variation += var
            cluster_sizes += c_sizes

        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        logging.info('%3d\t\t%f\t%f' % (j, total_variation, delta_variation))

        # Recompute centroids
        worker_args = [(lo, hi, k, data) for (lo, hi) in slices]
        ret_list = p.starmap(recomputeCentroids, worker_args)
        centroids = np.sum(ret_list, axis=0)
        centroids = centroids / cluster_sizes.reshape(-1, 1)

        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
    return total_variation, np.array(c)


def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s', level=logging.INFO)
    if args.debug: 
        logging.basicConfig(format='# %(message)s', level=logging.DEBUG)

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    total_variation_seq, assignment_seq = kmeansSerial(args.k_clusters,
                                                       X,
                                                       args.iterations,
                                                       args.workers)
    end_time = time.time()
    print('Serial clustering complete in %3.2f [s]' % (end_time - start_time))

    start_time = time.time()
    total_variation, assignment = kmeans(args.k_clusters, X, args.iterations,
                                         args.workers)
    end_time = time.time()
    print('Parallel clustering complete in %3.2f [s]' % (end_time - start_time))
    print(f'Total variation {total_variation}')
    logging.debug(f'Var diff: {total_variation_seq - total_variation}')
    logging.debug(f'K   diff: {np.allclose(assignment, assignment_seq)}')

    if args.plot: # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title('k-means result')
        plt.grid()
        #plt.show()
        fig.savefig(args.plot)
        plt.close(fig)


if __name__ == '__main__':
    '''
Problem 3 (4 pt):

Prepare the parallel implementation of the k-means algorithm using the
multi-processing Python package. Use the k-means implementation in kmeans.py
Download kmeans.py as the starting point.

a)  (2 pt) Describe the building blocks of the k-means algorithm, which building
blocks possibly can be parallelized, where and what data has to be exchanged,
and whether sequential bottlenecks (in the sense that they cannot be
parallelized) possibly exist. The description can be provided as a brief sketch
of the method in pseudo-code, or by annotating (e.g. drawing over) a listing of
kmeans.py.

b)  (1 pt) Measure the overall running time and the individual running times of
the sections of the serial program you intend to parallelize. Use at least
10,000 sampled points for k-means.

c)  (1 pt) For the two sections which contribute the most to the total running
time: Use Amdahl’s law to compute the theoretical total speedup assuming 4,
respectively 8, cores.
    '''
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
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
    computeClustering(args)