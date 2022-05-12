'''
Problem 3 (4 pt): Given keys, k1, k2, ...., kn and their frequencies f1, f2,
..., fn sample keys according to their frequencies. The probability of picking
any key is pi = fi / (Sum(fj)); let p = (p1, ..., pn). Sampling n keys according
to their frequencies is possible with the following code. Here keys is a numpy
array containing the keys and p a numpy array containing the corresponding
probabilities.

import numpy
rng = numpy.random.default_rng()
rng.choice(keys, n, replace=True, p=p)

Provide a more efficient, serial version of sampling from all the (keys,
frequency) pairs in /data/2022-DIT065-DAT470/Ensembl-cds-15-counts.txt. Report
the speed as samples per second just for the sampling, i.e., excluding time to
load the data.
'''
import argparse
import random
import time
import timeit
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    rng = np.random.default_rng()

    def numpy_random(keys, n, p, rng=rng):
        '''
        Syntactic-sugar to have the same interface for all functions
        '''
        return rng.choice(keys, n, replace=True, p=p)

    def serial_random(keys, n, p, cdf=None):
        '''
        It leverages the inverse transform sampling algorithm. Source and more
        information at: https://en.wikipedia.org/wiki/Inverse_transform_sampling

        Possible other method: Stochastic Collocation Monte Carlo sampler (SCMC
        sampler).
        '''
        cumsum = cdf if cdf is not None else np.cumsum(p)
        # Use inverse of the CDF function
        rdm_unif = np.random.rand(n)
        return keys[np.searchsorted(cumsum, rdm_unif)]

    def python_random(keys, n, p):
        '''
        Functionally correct and matching numpy implementation, but terribly
        slow, i.e. it utilizes two nested for-loops...
        '''
        sampled_keys = []
        cumsum = np.cumsum(p)
        for i in range(n):
            rand = random.uniform(0, 1)
            for j, pj in enumerate(cumsum):
                if rand < pj:
                    sampled_keys.append(keys[j])
                    break
        return np.array(sampled_keys)
    # ==========================================================================
    # Testing timing
    # ==========================================================================
    '''
    n_runs = 10000
    seq_t = timeit.timeit(lambda: serial_random(keys, n, p), number=n_runs)
    py_t = timeit.timeit(lambda: python_random(keys, n, p), number=n_runs)
    numpy_t = timeit.timeit(lambda: numpy_random(keys, n, p), number=n_runs)
    print(f'Average time Serial impl.: {seq_t / n_runs * 1000:.4f} ms')
    print(f'Average time Python impl.: {py_t / n_runs * 1000:.4f} ms')
    print(f'Average time Numpy impl.:  {numpy_t / n_runs * 1000:.4f} ms')
    '''
    # ==========================================================================
    # Checking correctness
    # ==========================================================================
    '''
    n = 6
    keys = np.random.randn(n)**2
    p = np.random.randn(n)**2
    p /= p.sum()
    n_runs = 10000
    py_sum = np.zeros(n)
    np_sum = np.zeros(n)
    cdf = np.cumsum(p)
    for _ in range(n_runs):
        for k in serial_random(keys, n, p, cdf):
            py_sum[np.where(k == keys)] += 1

        for k in numpy_random(keys, n, p):
            np_sum[np.where(k == keys)] += 1

    fig, ax = plt.subplots()
    bar_width = 0.35
    # Bar position
    seq_index = [x for x in range(n)]
    np_index = [x + bar_width for x in range(n)]
    tx_index = [x + bar_width / 2 for x in range(n)]
    # Plot bars
    plt.bar(seq_index, py_sum, bar_width, label='Serial')
    plt.bar(np_index, np_sum, bar_width, label='Numpy')
    # Plot values on top of bars
    for i in range(n):
        x_seq = i - bar_width / 4
        x_np = i + bar_width - bar_width / 4
        ax.text(x_seq, py_sum[i], f'{py_sum[i]:.1f}')
        ax.text(x_np, np_sum[i], f'{np_sum[i]:.1f}')
    plt.xticks(tx_index, [f'{i}' for i in p])
    plt.xlabel('p')
    plt.ylabel('Number of samples')
    plt.title('Checking np.choice and serial implementation')
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(filename)
    plt.show()
    '''
    # ==========================================================================
    # Read keys, frequency pairs from file and compute p
    # ==========================================================================
    # TODO(Stefano): The results from reading the frequencies from the file
    # shouldn't differ much from the random tests I've performed...

    # ==========================================================================
    # Measuring timing for different sample sizes
    # ==========================================================================
    n_runs = 10000
    n_values = []
    seq_time = []
    np_time = []
    for n in range(0, 6000, 500):
        if n == 0:
            continue
        print(f'Measuring n = {n}')
        # Init keys and p values
        keys = np.random.randn(n)**2
        p = np.random.randn(n)**2
        p /= p.sum()
        # Measure time
        cdf = np.cumsum(p) # NOTE: Precompute the CDF
        seq_t = timeit.timeit(lambda: serial_random(keys, n, p, cdf), number=n_runs)
        np_t = timeit.timeit(lambda: numpy_random(keys, n, p), number=n_runs)
        # Collect sample rate
        n_values.append(n)
        seq_time.append(n / seq_t) # seq_t / n_runs * 1000000)
        np_time.append(n / np_t) # np_t / n_runs * 1000000)
    # Plotting
    use_bar_plot = False
    if not use_bar_plot:
        plt.plot(n_values, seq_time, '-o', label='Serial')
        plt.plot(n_values, np_time, '-d', label='Numpy')
    else:
        fig, ax = plt.subplots()
        bar_width = 0.35
        # Bar position
        seq_index = [x for x in range(len(n_values))]
        np_index = [x + bar_width for x in range(len(n_values))]
        tx_index = [x + bar_width / 2 for x in range(len(n_values))]
        # Plot bars
        plt.bar(seq_index, seq_time, bar_width, label='Serial')
        plt.bar(np_index, np_time, bar_width, label='Numpy')
        # Plot values on top of bars
        for i in range(len(n_values)):
            x_seq = i - bar_width / 4
            x_np = i + bar_width - bar_width / 4
            ax.text(x_seq, seq_time[i], f'{seq_time[i]:.1f}')
            ax.text(x_np, np_time[i], f'{np_time[i]:.1f}')
        plt.xticks(tx_index, [f'{n}' for n in n_values])
    plt.xlabel('Sample size n')
    plt.ylabel('Sample Rate [sample/s]')
    plt.title('Numpy vs. Serial Implementation Timing')
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(filename)
    plt.show()


if '__main__':
    sample_file = '/data/2022-DIT065-DAT470/Ensembl-cds-15-counts.txt'
    program_description = 'Improving Numpy sampling function.'

    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('--filename',
                        default=sample_file,
                        type=str,
                        help=f'Text file to process. Default: {sample_file}')
    args = parser.parse_args()
    main(args)
