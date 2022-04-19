from mrjob.job import MRJob
from matplotlib.pyplot import hist
from time import process_time
import numpy as np

class Summary(MRJob):
    def mapper(self, _, line):
        _, _, number = line.split()
        yield 'value', float(number)

    def reducer(self, _, numbers):
        numbers = np.fromiter(numbers, dtype=float)
        yield ('median', np.median(numbers))

if __name__ == '__main__':
    start = process_time()
    Summary.run()
    processing_time = process_time() - start
    print('time: {} s'.format(processing_time))