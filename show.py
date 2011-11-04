import sys
import numpy as np
from matplotlib import pyplot as plt

def show_times(d, r, kcolor='b', rcolor='r'):
    start = np.min(d[:, 3:])

    # get time data and subtract the huge starting number
    sk = d[d[:,2] == 0][:,3:] - start
    for i in range(r):
        plt.plot([sk[i][0], sk[i][2]], [0.0, 1.0], '%s--' % kcolor)
        plt.plot([sk[i][2], sk[i][3]], [1.0, 1.0], '%so-' % kcolor)

    sr = d[d[:,2] == 1][:,3:] - start
    for i in range(r):
        plt.plot([sr[i][0], sr[i][2]], [0.0, -1.0], '%s--' % rcolor)
        plt.plot([sr[i][2], sr[i][3]], [-1.0, -1.0], '%so-' % rcolor)


if __name__ == '__main__':
    converters = { 3:int, 4:int, 5:int, 6:int }
    try:
        data = np.loadtxt(sys.argv[1], dtype=np.long, delimiter=' ', converters=converters)
    except:
        sys.stderr.write('Could not open trace file\n')
        sys.exit(1)
    single = data[data[:,0] == 0]
    multi = data[data[:,0] == 1]

    # evaluate single GPU
    plt.subplot(211)
    show_times(single, 5)

    # handle multi gpu case
    m1 = multi[multi[:,1] == 0]
    m2 = multi[multi[:,1] == 1]

    plt.subplot(212)
    show_times(m1, 5)
    show_times(m2, 5, 'c', 'y')
    

    plt.show()
