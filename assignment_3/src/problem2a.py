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
 
        yield ('minimum', np.min(numbers))
        yield ('maximum', np.max(numbers))
        yield ('mean', np.mean(numbers))
        yield ('standard_deviation', np.std(numbers))

        bins, histogram, _ = hist(numbers, bins=10)
        yield ('hist', (bins.tolist(), histogram.tolist()))

if __name__ == '__main__':
    start = process_time()
    Summary.run()
    processing_time = process_time() - start
    print('time: {} s'.format(processing_time))
