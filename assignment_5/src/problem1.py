"""
Problem 1 (5 pt): To decide the amount of memory needed for storing all words
(i.e. strings only containing the letters a-z) appearing in a large text in a
trie, compute the average length of shared prefixes between consecutive words in
a sorted list of unique words from the text. For example for the text
"Barney's barn is burning", the sorted list of words consists of barn, barneys
(removing any characters not in a-z), burning, is (note that case is ignored)
and the shared prefixes are barn, b, "" with an average of (4+1+0) / 3, in
particular: {len('barn'): 4, len('b'): 1, len(''): 0}. Implement a solution
using PySpark (or MrJob at your discretion) returning the average length of the
prefixes, the total number of unique words and the average word length. Hint: Do
not use use a sort on all words in the text (or external sort programs).  There
is text data on all machines available at /data/2022-DIT065-DAT470/gutenberg/,
e.g. /data/2022-DIT065-DAT470/gutenberg/060/06049.txt.
"""
import math
from math import sqrt
from mrjob.job import MRJob, MRStep
from collections import namedtuple
from operator import itemgetter

Point = namedtuple('Point', 'x y')

def nearestCentroid(datum, centroids):
    # dist = np.linalg.norm(centroids - datum, axis=1)
    # return np.argmin(dist), np.min(dist)
    dists = []
    for c in centroids:
        dists.append(sqrt((datum.x - c.x)**2 + (datum.y - c.y)**2))
    return min(enumerate(dists), key=itemgetter(1))

def loadCentroids(filename):
    centroids = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = line.split('\t')
            centroids.append([float(x), float(y)])
    return centroids

class KMeans(MRJob):

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init, mapper=self.mapper,
                   reducer=self.reducer),
        ]

    def configure_args(self):
        super(KMeans, self).configure_args()
        self.add_file_arg('--centroids',
            help='File containing centroid values, one per line.')

    def mapper_init(self):
        filename = self.options.centroids
        self.centroids = [Point(*c) for c in loadCentroids(filename)]

    def mapper(self, _, line):
        # Returns tuples: <cluster, point>
        x, y = line.split('\t')
        x, y = float(x), float(y)
        cluster, _ = nearestCentroid(Point(x, y), self.centroids)
        yield cluster, (x, y)

    def reducer(self, key, values):
        # Receives tuples: <cluster, list of points belonging to it>, meaning
        # that for each centroid, i.e. key, there's a list of data values to
        # average.
        centroid_x, centroid_y = 0, 0
        for cluster_size, (datum_x, datum_y) in enumerate(values):
            centroid_x += datum_x
            centroid_y += datum_y
        centroid_x = centroid_x / (cluster_size + 1)
        centroid_y = centroid_y / (cluster_size + 1)
        # yield key, (centroid_x, centroid_y)
        yield centroid_x, centroid_y


if __name__ == '__main__':
    # Assume that the data comes from stdin, which is default for MRJob
    KMeans.run()