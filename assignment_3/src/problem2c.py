from mrjob.job import MRJob
from time import process_time


class Summary(MRJob):
    SORT_VALUES = True

    def mapper(self, _, line):
        _, _, number = line.split()
        yield 'value', float(number)

    def combiner(self, key, numbers):
        # numbers = np.fromiter(numbers, dtype=float)
        numbers = tuple(numbers)
        yield ('median', numbers[len(numbers) // 2])

    def reducer(self, _, numbers):
        # numbers = np.fromiter(numbers, dtype=float)
        numbers = tuple(numbers)
        yield ('median', numbers[len(numbers) // 2])


if __name__ == '__main__':
    start = process_time()
    Summary.run()
