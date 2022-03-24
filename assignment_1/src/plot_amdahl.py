import argparse
import logging
from timeit import timeit
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_theoretical_sp(k):
    return 1


def time_compute_pi(k, s):
    cmd = f'python ./mp_pi_montecarlo_pool.py -w {k} -s {s}'.split(' ')
    proc = Popen(cmd, stderr=PIPE, stdout=PIPE)
    proc.communicate()


def plot_amdhal(args):
    k_values = [2, 4, 8, 16, 32]
    theory_speedups = []
    actual_speedups = []
    steps = args.steps
    # Get experimental results
    logger.info(f'Calculating baseline for k=1 with {steps} steps')
    base_perf = timeit(stmt=f'time_compute_pi(1, {steps})',
                       setup='from __main__ import time_compute_pi',
                       number=100)
    logger.info(f'Baseline performance was {base_perf} for k=1')
    for k in k_values:
        logger.info(f'Executing for k = {k} processes with {steps} steps')
        theory_sp = get_theoretical_sp(k)
        time_perf = timeit(stmt=f'time_compute_pi({k}, {args.steps})',
                           setup='from __main__ import time_compute_pi',
                           number=100)
        logger.info(f'k = {k} with {steps} took {time_perf}')
        actual_sp = base_perf / time_perf
        theory_speedups.append(theory_sp)
        actual_speedups.append(actual_sp)

    # Bar plot template
    fig, ax = plt.subplots()
    bar_width = 0.35 * 2
    index = [x + bar_width for x in range(len(theory_speedups))]
    plt.bar(index + bar_width, (theory_speedups), bar_width, zorder=2,
            color='green', label='Theoretical Sp')
    plt.xticks(index, k_values, rotation=-90)
    # TODO(Stefano): Add measured speedup
    # ...
    plt.ylabel('Speedup')
    plt.title('Theoretical vs. Measured Speedup')
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.tight_layout()
    # plt.savefig('amdhal_sp.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Amdhal Speedup of computing Pi using MC simulation.')
    parser.add_argument('--repeats', '-r',
                        default='100',
                        type = int,
                        help='How many times to repeat for each amount of processes')
    parser.add_argument('--steps', '-s',
                        default='1000',
                        type = int,
                        help='Number of steps in the Monte Carlo simulation')
    args = parser.parse_args()
    plot_amdhal(args)