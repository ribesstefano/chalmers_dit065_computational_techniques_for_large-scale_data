
import math

from mrjob.job import MRJob, MRStep

BINS = 10


class Means(MRJob):

    SORT_VALUES = True

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.minmax_reducer),
            MRStep(mapper=self.final_mapper, reducer=self.output_reducer),
            MRStep(reducer=self.bin_output_reducer)
        ]

    def mapper(self, _, line):
        _, _, val = line.split('\t')
        # For min/max we need to process on the same node, so use None as a key
        yield None, float(val)

    def minmax_reducer(self, _, values):
        try:
            memory_vals = tuple(values)
            min_ = memory_vals[0]
            max_ = memory_vals[-1]
            elems = len(memory_vals)
            mean = sum(memory_vals) / elems
            for val in memory_vals:
                # val as key, because it's a float, and we don't really care which node handles it
                yield val, (min_, max_, elems, mean)
        except ValueError as e:
            raise e

    def final_mapper(self, val, value):
        min_, max_, elems, mean = value
        std_ = (val - mean) ** 2

        # Work out which bin the value goes in
        # TODO: Can we only do this calculation once?
        range = max_ - min_
        bin_size = range / BINS
        bin_ = math.floor((val - min_)/ bin_size)

        yield None, (val, min_, max_, bin_, elems, mean, std_)

    def output_reducer(self, _, values):
        i = 0
        elems = 0
        sum_ = 0
        for value in values:
            val, min_, max_, bin_, elems, mean, std_ = value
            # We only need to care about the descriptive statistics once
            if i == 0:
                yield "min", min_
                yield "max", max_
                yield "mean", mean
                i += 1
            sum_ += std_
            # Now we just need to yield the correct bin so we can count them in the next reducer
            yield bin_, 1
        yield "std_dev", math.sqrt(sum_ / elems)

    def bin_output_reducer(self, key, values):
        if key in ("min", "max", "mean", 'std_dev'):
            yield key, next(values)
        else:
            yield f"bin_{key}", sum(values)


if __name__ == '__main__':
    # Assume that the data comes from stdin, which is default for MRJob
    from time import process_time

    start = process_time()
    out = Means.run()
    processing_time = process_time() - start
    # TODO (Huw): Printing processing time seems to interfere with processing
    # print(processing_time)
