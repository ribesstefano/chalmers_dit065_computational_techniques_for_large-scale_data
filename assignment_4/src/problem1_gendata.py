import argparse
from sklearn.datasets import make_blobs


def generateData(n, c):
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X


def main(args):
    X = generateData(args.samples, args.classes)
    with open(args.filename, 'w') as f:
        for x in X:
            f.write(f'{x[0]}\t{x[1]}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: problem1_gendata.py -с 4 --samples 10000 --filename kmeans.data'
    )
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')
    parser.add_argument('--samples', '-s',
                        default='10000',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--filename', '-f',
                        default='kmeans.dat',
                        type = str,
                        help='K-means data point filename')
    args = parser.parse_args()
    main(args)