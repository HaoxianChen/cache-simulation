import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse
from cache import *


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('s0', type=int, help='bgin cache size')
parser.add_argument('s1', type=int, help='end cache size')
parser.add_argument("-s","--step", type=int, help='simulation step in cache\
sizes', default = 4)
args = parser.parse_args()

# simulation parameters
x1 = 16 # size of array 1
x2 = 48 # size of array 2
p1 = 0.7 # probability of accessing array 1
p2 = 0.3 # probability of accessing array 2
iterate_times = 5000
ed = x1 + x2 # overall expected reuse distance: the working set size, also equals
assert p1 + p2 == 1

d1 = x1/p1
d2 = x2/p2

# policy: give as values of each age
value = np.ones(MAX_AGE,dtype=float)
value[0] = 0

array1= range(0,x1)
array2 = range(x1,x1+x2)
cache_size = range(args.s0,args.s1,args.step)
miss_rate = np.zeros(len(cache_size),)

test_spec = str(args.s0) + '-' + str(args.s1) + '-' + str(args.step)
LOG_FILENAME = 'logs/trace-' + test_spec +'.log'
logging.basicConfig(filename=LOG_FILENAME, filemode='w+',level=logging.DEBUG)
logger = logging.getLogger(__name__)

FILENAME = 'logs/curve-' + test_spec +'.log'
f = open(FILENAME,'w')
f.write(str(len(cache_size))+'\n')

def weighted_choice(weights):
    totals = []
    running_total = 0
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total

    for i,total in enumerate(totals):
        if rnd < total:
            return i

for j,s in enumerate(cache_size):
    logger.info('simulating cache size at ' + str(s))

    cache = Cache(s,value)
    a_counter1 = 0
    a_counter2 = 0
    for i in range(iterate_times):
        # simulate data access
        k = weighted_choice([p1*100,p2*100])
        if k == 0:
            addr = array1[a_counter1 % len(array1)]
            a_counter1 += 1
        elif k == 1:
            addr = array2[a_counter2 % len(array2)]
            a_counter2 += 1
        cache.lookup(addr)
    miss_rate[j] = 1-cache.get_hit_rate()
    print 'cache size:' + str(s) + ' miss rate:' + str(miss_rate[j])

    # log hit age and eviction age distribution
    f.write(str(s)+'\n')
    for a in cache.get_hit_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')
    for a in cache.get_evict_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')

    # log age values
    for v in value:
        f.write(str(v)+' ')
    f.write('\n')

# log miss rate curve
for s in cache_size:
    f.write(str(s)+' ')
f.write('\n')
for r in miss_rate.tolist():
    f.write(str(r)+' ')
f.write('\n')
f.close()
