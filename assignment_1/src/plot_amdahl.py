from mp_pi_montecarlo_pool import compute_pi

import matplotlib.pyplot as plt

def get_theoretical_sp(k):
    return 1

def time_compute_pi(k):
    return 1

def plot_amdhal(args):
    k_values = [2, 4, 8, 16, 32]
    theory_speedups = []
    actual_speedups = []
    # Get experimental results
    base_perf = time_compute_pi(k=1)
    for k in k_values:
        theory_sp = get_theoretical_sp(k)
        time_perf = time_compute_pi(k=1)
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
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes')
    parser.add_argument('--steps', '-s',
                        default='1000',
                        type = int,
                        help='Number of steps in the Monte Carlo simulation')
    args = parser.parse_args()
    plot_amdhal(args)