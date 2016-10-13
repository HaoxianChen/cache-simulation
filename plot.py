import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename',type=str)
parser.add_argument('gtype',type=str)
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                            help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                            const=sum, default=max,
#                                               help='sum the integers (default:\
#                                               find the max)')
args = parser.parse_args()

f = open(args.filename)
data_len = int(f.readline())

def model(cache_size):
    p = 0.5
    D1 = 32
    y = np.zeros(len(cache_size),)
    for i,s in enumerate(cache_size):
        y[i] = ((1-p) * s*s - (2-p)*D1*s + D1*D1)/(D1*D1 - D1*s)
    return y

for i in range(data_len):
    size  = f.readline()
    line = f.readline()
    line2 = f.readline()
    if args.gtype == 'h':
        y = np.fromstring(line,dtype=float,sep=' ')
        plt.subplot((data_len*2+1)/2,2,i*2+1)
        plt.subplots_adjust(hspace=.8)
        plt.plot(y)
        plt.xlabel('age')
        plt.title('hit age distribution at cache size ' +size)
        y = np.fromstring(line2,dtype=float,sep=' ')
        plt.subplot((data_len*2+1)/2,2,i*2+2)
        plt.plot(y)
        plt.xlabel('age')
        plt.title('evict age distribution at cache size ' +size)

    line = f.readline()
    if args.gtype == 'r':
        x = np.fromstring(line,dtype=float,sep=' ')
        plt.subplot(data_len,1,i)
        plt.title('cache size: ' + str(size))
        plt.plot(x,label='age values')
        plt.axis([0,len(x),min(x[0:80]),max(x)])
        plt.xlabel('ages')
        plt.ylabel('value')

if args.gtype == 'm':
    line = f.readline()
    x = np.fromstring(line,dtype=float,sep=' ')
    line = f.readline()
    y = np.fromstring(line,dtype=float,sep=' ')
    y2 = model(x)
    plt.figure()
    plt.plot(x,y,label='simulation')
    plt.plot(x,y2,label='model')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('cache size')
    plt.ylabel('miss rate')
    plt.axis([0,64,0,1])

plt.show()
