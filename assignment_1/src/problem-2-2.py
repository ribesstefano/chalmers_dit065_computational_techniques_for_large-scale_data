import argparse
import logging
import math
import matplotlib.pyplot as plt
from timeit import timeit
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_theoretical_sp(p, s):
    MUL_CYCLES = 7
    DIV_CYCLES = 12
    T_par = s * (2 * MUL_CYCLES + 3 + math.pi / 4)
    T_seq = s + 1 + DIV_CYCLES
    return (T_seq + T_par) / (T_seq + T_par / p)


def time_compute_pi(p, s):
    cmd = f'python3 ./mp-pi-montecarlo-pool.py -w {p} -s {s}'.split(' ')
    subprocess.run(cmd, shell=False)


def plot_amdhal(args):
    # TODO(Stefano): It should be made generic enough to be re-usable in
    # future assignments. Maybe having the function/process/script to launch as
    # an additional argument?
    p_values = [2, 4, 8, 16, 32]
    theory_speedups = []
    actual_speedups = []
    steps = args.steps
    # Get experimental results
    logger.info(f'Calculating baseline for p = 1 with {steps} steps')
    base_perf = timeit(stmt=f'time_compute_pi(1, {steps})',
                       setup='from __main__ import time_compute_pi',
                       number=args.repeats)
    logger.info(f'Baseline performance was {base_perf:.3f} ms for p = 1')
    for p in p_values:
        logger.info(f'Executing for p = {p} workers with {steps} steps')
        theory_sp = get_theoretical_sp(p, args.steps)
        time_perf = timeit(stmt=f'time_compute_pi({p}, {steps})',
                           setup='from __main__ import time_compute_pi',
                           number=args.repeats)
        # TODO(Stefano): Is it measured in ms? Or s?
        logger.info(f'k = {p} with {steps} took {time_perf:.3} ms')
        actual_sp = base_perf / time_perf
        theory_speedups.append(theory_sp)
        actual_speedups.append(actual_sp)

    # Bar plot template
    fig, ax = plt.subplots()
    bar_width = 0.35
    th_index = [x for x in range(len(p_values))]
    ac_index = [x + bar_width for x in range(len(p_values))]
    tx_index = [x + bar_width / 2 for x in range(len(p_values))]
    plt.bar(th_index, theory_speedups, bar_width, label='Theoretical Sp')
    plt.bar(ac_index, actual_speedups, bar_width, label='Measured Sp')
    # Plot values on top of bars
    for i in range(len(p_values)):
        x_theory = i - bar_width / 4
        x_actual = i + bar_width - bar_width / 4
        ax.text(x_theory, theory_speedups[i], f'{theory_speedups[i]:.1f}')
        ax.text(x_actual, actual_speedups[i], f'{actual_speedups[i]:.1f}')
    plt.xticks(tx_index, [f'p = {p}' for p in p_values])
    plt.xlabel('Number of cores p')
    plt.ylabel('Speedup')
    plt.title('Theoretical vs. Measured Speedup')
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('amdhal_sp_theoretical.pdf')
    # plt.show()


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