import logging
import time
import multiprocessing # See https://docs.python.org/3/library/multiprocessing.html
import numpy as np
import matplotlib.pyplot as plt
from problem1 import generateData, kmeansSerial, nearestCentroid
from collections import namedtuple

def amdhals_law(f, w):
    return 1 / (1 - f + f / w)


def plot_amdhal(w_values, actual_speedups, theory_speedups, filename=None,
                show=False, theory_label='Theoretical Sp',
                plot_title='Theoretical vs. Measured Speedup'):
    # Bar plot template
    fig, ax = plt.subplots()
    bar_width = 0.35
    th_index = [x for x in range(len(w_values))]
    ac_index = [x + bar_width for x in range(len(w_values))]
    tx_index = [x + bar_width / 2 for x in range(len(w_values))]
    plt.bar(th_index, theory_speedups, bar_width, label=theory_label)
    plt.bar(ac_index, actual_speedups, bar_width, label='Measured Sp')
    # Plot values on top of bars
    for i in range(len(w_values)):
        x_theory = i - bar_width / 4
        x_actual = i + bar_width - bar_width / 4
        ax.text(x_theory, theory_speedups[i], f'{theory_speedups[i]:.1f}')
        ax.text(x_actual, actual_speedups[i], f'{actual_speedups[i]:.1f}')
    plt.xticks(tx_index, [f'w = {w}' for w in w_values])
    plt.xlabel('Number of workers w')
    plt.ylabel('Speedup')
    plt.title(plot_title)
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

NUM_SAMPLES = 10000
K_CLUSTERS = 4
CLASSES = 4
ITERATIONS = 100
PARALLEL_f = 0.999

global X
global c
X = generateData(NUM_SAMPLES, CLASSES)
c = multiprocessing.Array('i', [0] * len(X))


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


def main():
    workers = [1, 2, 4, 8, 16, 32]
    actual_speedups = []
    theory_speedups = []

    # X = generateData(NUM_SAMPLES, CLASSES)
    # global c
    # c = multiprocessing.Array('i', [0] * len(X))

    for w in workers:
        print(f'INFO. Running K-Means with {w} workers.')
        start_time = time.time()
        total_variation_seq, assignment_seq = kmeansSerial(K_CLUSTERS, X,
                                                           ITERATIONS, w)
        end_time = time.time()
        serial_t = end_time - start_time

        start_time = time.time()
        total_variation, assignment = kmeans(K_CLUSTERS, X, ITERATIONS, w)
        end_time = time.time()
        parallel_t = end_time - start_time

        actual_speedups.append(serial_t / parallel_t)
        theory_speedups.append(amdhals_law(PARALLEL_f, w))

    filename = 'problem1.pdf'
    plot_amdhal(workers, actual_speedups, theory_speedups, filename=filename,
                show=True, theory_label='Theoretical Sp',
                plot_title='Theoretical vs. Measured Speedup')


if __name__ == '__main__':
    main()