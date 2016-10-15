import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename',type=str)
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                            help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                            const=sum, default=max,
#                                               help='sum the integers (default:\
#                                               find the max)')
args = parser.parse_args()

f = open(args.filename)
data_len = int(f.readline())

for i in range(data_len):
    line = f.readline()
    values = np.fromstring(line,dtype=float,sep=' ')
    plt.subplot(data_len, 1, i+1)
    plt.plot(values)
    plt.axis([0,len(values),min(values[0:90]),max(values)])
    plt.title(str(i))

plt.show()
