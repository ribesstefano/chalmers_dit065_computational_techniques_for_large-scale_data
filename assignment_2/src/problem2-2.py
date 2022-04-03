import datetime
import multiprocessing  # See https://docs.python.org/3/library/multiprocessing.html
import argparse  # See https://docs.python.org/3/library/argparse.html
import random
from math import pi


def sample_pi(init):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
    :param init: A tuple or an int.  If int, the amount of steps to run for.
    If a tuple, should be (int, int) and be amount of steps, and seed.
    :return: the number of sucesses."""
    # For backwards compatibility check for the type of the tuple
    if isinstance(init, tuple):
        n, seed = init
        assert isinstance(seed, int), 'Seed must be an integer'
        random.seed(seed)
    else:
        n = init
        random.seed()
    s = 0
    assert isinstance(n, int), 'Steps must be an integer'
    for i in range(n):
        x = random.random()
        y = random.random()
        if x ** 2 + y ** 2 <= 1.0:
            s += 1
    return s


def worker(tasks, results):
    """Runs an instance of sample_pi and returns the result to results queue

    :param tasks: A JoinableQueue which will contain the tasks to be performed
    :param results: A Queue to return the result of sample_pi
    """
    while True:
        init = tasks.get()
        if init is None:
            break
        ret = sample_pi(init)
        results.put(ret)
        tasks.task_done()


def compute_pi(args):
    start = datetime.datetime.now()
    accuracy = args.accuracy
    # print(accuracy)
    seed = args.seed
    random.seed(seed) if seed is not None else random.seed()

    n = args.steps or 50
    results = multiprocessing.Queue()
    tasks = multiprocessing.JoinableQueue()

    p = multiprocessing.Pool(args.workers, worker, (tasks, results,))

    def finish(n, s):
        """Decides whether the simulation is accurate enough to end

        :param n: The amount of steps that have been performed
        :param s: The amount of successes
        :return: True if we should finish, False if not
        """
        if n == 0:
            return False
        pi_est = (4.0 * s) / n
        if abs(pi_est - pi) <= accuracy:
            return True
        return False

    n_total = 0
    s_total = 0

    # print(" Steps\tSuccess\tPi est.\tError")

    while not finish(n_total, s_total):
        proc_args = [(n, random.randint(0, 2 ** 16))
                     for i in range(args.workers)]

        # Run the simulation again
        for a in proc_args:
            tasks.put(a)

        # Wait for all workers to finish, then update the current results
        tasks.join()
        while not results.empty():
            s_total += results.get()
            n_total += n
            pi_est = (4.0 * s_total) / n_total
            # print("%6d\t%7d\t%1.5f\t%1.5f" %
            #       (n_total, s_total, pi_est, pi - pi_est))

    # Break out of their loops
    for i in range(args.workers):
        tasks.put(None)

    p.close()
    end = datetime.datetime.now()

    print(n_total // n, (end - start).total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Pi using Monte Carlo simulation.')
    parser.add_argument('--workers', '-w',
                        default='4',
                        type=int,
                        help='Number of parallel processes')
    parser.add_argument('--steps', '-s',
                        default='50',
                        type=int,
                        help='Number of steps in the Monte Carlo simulation for each process')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Seed for the PRNG. If None takes system default which is not repeatable')
    parser.add_argument('--accuracy', '-a',
                        default=0.0001,
                        type=float,
                        help='The accuracy at which to stop the simulation. Defaults to 0.0001')
    args = parser.parse_args()
    compute_pi(args)
