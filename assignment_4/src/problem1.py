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