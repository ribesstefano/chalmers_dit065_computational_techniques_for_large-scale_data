import multiprocessing  # See https://docs.python.org/3/library/multiprocessing.html
import argparse  # See https://docs.python.org/3/library/argparse.html
import random
from math import pi


def sample_pi(init):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of sucesses."""
    n, seed = init
    random.seed(seed)
    print("Hello from a worker")
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x ** 2 + y ** 2 <= 1.0:
            s += 1
    return s


def compute_pi(args):
    seed = args.seed
    random.seed(seed) if seed is not None else random.seed()
    n = int(args.steps / args.workers)

    p = multiprocessing.Pool(args.workers)
    seeds = [(n, random.randint(0, 2 ** 16))
             for n, seed in enumerate(range(args.workers))]
    print(seeds)
    s = p.map(sample_pi, seeds)

    n_total = n * args.workers
    s_total = sum(s)
    pi_est = (4.0 * s_total) / n_total
    print(" Steps\tSuccess\tPi est.\tError")
    print("%6d\t%7d\t%1.5f\t%1.5f" % (n_total, s_total, pi_est, pi - pi_est))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers', '-w',
                        default='1',
                        type=int,
                        help='Number of parallel processes')
    parser.add_argument('--steps', '-s',
                        default='1000',
                        type=int,
                        help='Number of steps in the Monte Carlo simulation')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Seed for the RNG. If None takes system default which is not repeatable')
    args = parser.parse_args()
    compute_pi(args)
