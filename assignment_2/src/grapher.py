import argparse
import datetime
import logging
import math
import subprocess
import random
import matplotlib.pyplot as plt
from timeit import timeit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def amdhals_law(T_seq, T_par, p):
    return (T_seq + T_par) / (T_seq + T_par / p)


def plot_amdhal(p_values, actual_speedups, theory_speedups, filename=None,
                show=False, theory_label='Theoretical Sp',
                plot_title='Theoretical vs. Measured Speedup'):
    # Bar plot template
    fig, ax = plt.subplots()
    bar_width = 0.35
    th_index = [x for x in range(len(p_values))]
    ac_index = [x + bar_width for x in range(len(p_values))]
    tx_index = [x + bar_width / 2 for x in range(len(p_values))]
    plt.bar(th_index, theory_speedups, bar_width, label=theory_label)
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
    plt.title(plot_title)
    plt.grid(which='both', axis='y', alpha=0.7, zorder=1)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


def time_compute_pi(p, accuracy=0.0001):
    cmd = f'python3 ./problem2-2.py -w {p} -a {accuracy} -s 200 --seed 42'.split(' ')
    p = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode('utf-8')
    return out.split(' ')


def compute_pi_par_part(steps):
    random.seed(1)
    random.seed()
    s = 0
    for i in range(steps):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s


def compute_pi_seq_part(steps, s):
    return (4.0 * s) / steps


def serial_compute_pi(steps):
    s = compute_pi_par_part(steps)
    return compute_pi_seq_part(steps, s)


def get_theoretical_sp(p, steps):
    MUL_CYCLES = 7
    DIV_CYCLES = 12
    T_par = steps * (2 * MUL_CYCLES + 3 + math.pi / 4)
    T_seq = 1 + DIV_CYCLES
    return amdhals_law(T_seq, T_par, p)


def run_experiments(args):
    # TODO(Stefano): It should be made generic enough to be re-usable in
    # future assignments. Maybe having the function/process/script to launch as
    # an additional argument?
    p_values = [2, 4, 8, 16, 32]
    empirical_speedups = []
    theory_speedups = []
    actual_speedups = []
    # Measure parallel and sequential fractions of time in order to calculate
    # Amdhal's law.
    # NOTE: We aim at calculating fractions of time, so any fairly large number
    # of steps should give meaningful results.
    steps = 1000000
    logger.info(f'Calculating empirical speedups. Timing parallel part.')
    time_par = timeit(stmt=f'compute_pi_par_part({steps})',
                       setup='from __main__ import compute_pi_par_part',
                       number=args.repeats)
    logger.info(f'Calculating empirical speedups. Timing sequential part.')
    time_seq = timeit(stmt=f'compute_pi_seq_part({steps}, {math.pi / 4 * steps})',
                       setup='from __main__ import compute_pi_seq_part',
                       number=args.repeats)
    logger.info(f'Calculating empirical speedups. Timing total execution time.')
    time_tot = timeit(stmt=f'serial_compute_pi({steps})',
                       setup='from __main__ import serial_compute_pi',
                       number=args.repeats)
    T_par = time_par / time_tot
    T_seq = time_seq / time_tot
    logger.info(f'`T_par: {T_par:.4f},\tT_seq: {T_seq:.4f}')
    # Get experimental results
    steps = args.steps
    accuracy = 0.0001
    logger.info(f'Calculating baseline for p = 1 with {accuracy} accuracy')

    def calculate_time(p):
        tuples_list = []
        for i in range(args.repeats):
            out = time_compute_pi(p, accuracy)
            tuples_list.append(out)
            # print(tuples_list)
        per_sec = [int(samples)/float(time_taken) for samples, time_taken in tuples_list]
        return sum(per_sec) / len(per_sec)

    base_perf = calculate_time(1)
    # base_perf = timeit(stmt=f'time_compute_pi(1, {steps})',
    #                    setup='from __main__ import time_compute_pi',
    #                    number=args.repeats)
    logger.info(f'Baseline performance was {int(base_perf)} samples for p = 1')
    for p in p_values:
        logger.info(f'Executing for p = {p} workers with accuracy {accuracy} and repeats {args.repeats}')
        theory_sp = get_theoretical_sp(p, args.steps)
        samples_list = []
        time_list = []
        samples_perf = calculate_time(p)
        # samples_perf = timeit(stmt=f'time_compute_pi({p}, {steps})',
        #                    setup='from __main__ import time_compute_pi',
        #                    number=args.repeats)

        # TODO(Stefano): Is it measured in ms? Or s?
        logger.info(f'k = {p} got {int(samples_perf)} samples/s')
        # We expect the samples to be higher than the  previous one
        actual_sp = samples_perf / base_perf
        theory_speedups.append(theory_sp)
        actual_speedups.append(actual_sp)
        empirical_speedups.append(amdhals_law(T_seq, T_par, p))

    plot_amdhal(p_values, actual_speedups, theory_speedups,
                filename='amdhal_sp_theoretical.pdf')
    plot_amdhal(p_values, actual_speedups, empirical_speedups,
                theory_label='Empirical Sp',
                plot_title='Empirical vs. Measured Speedup',
                filename='amdhal_sp_empirical.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Amdhal Speedup of computing Pi using MC simulation.')
    parser.add_argument('--repeats', '-r',
                        default='10',
                        type = int,
                        help='How many times to repeat for each amount of processes')
    parser.add_argument('--steps', '-s',
                        default='1000',
                        type = int,
                        help='Number of steps in the Monte Carlo simulation')
    args = parser.parse_args()
    run_experiments(args)
    # serial_compute_pi(args.steps)
