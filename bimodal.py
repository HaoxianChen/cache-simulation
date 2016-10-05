import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse
from cache import *

MAX_AGE = 150

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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('s0', type=int, help='bgin cache size')
parser.add_argument('s1', type=int, help='end cache size')
parser.add_argument("-s","--step", type=int, help='simulation step in cache\
sizes', default = 4)
parser.add_argument("-r","--showrank", type=int, help='show rank function')
args = parser.parse_args()

small_array = range(0,16)
big_array = range(16,64)
iterate_times = 10000
cache_size = range(args.s0,args.s1,args.step)
miss_rate = np.zeros(len(cache_size),)

test_spec = str(args.s0) + '-' + str(args.s1) + '-' + str(args.step)
LOG_FILENAME = 'logs/trace-' + test_spec +'.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logger = logging.getLogger(__name__)

FILENAME = 'logs/curve-' + test_spec +'.log'
f = open(FILENAME,'w')
f.write(str(len(cache_size))+'\n')

for j,s in enumerate(cache_size):
    logger.info('simulating cache size at ' + str(s))

    cache = Cache(s)
    k = weighted_choice([50,50])
    small_array_counter = 0
    big_array_counter = 0
    for i in range(iterate_times):
        if i % 2 == 0:
            k = weighted_choice([149,0])
        elif i % 2 == 1:
            k = weighted_choice([0,149])

        if k == 0:
            addr = small_array[small_array_counter % len(small_array)]
            small_array_counter += 1
        elif k == 1:
            addr = big_array[big_array_counter % len(big_array)]
            big_array_counter += 1
        cache.lookup(addr)
    miss_rate[j] = 1-cache.get_hit_rate()
    f.write(str(s)+'\n')
    for a in cache.get_hit_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')
    for a in cache.get_evict_ages().tolist():
        f.write(str(a)+' ')
    f.write('\n')


for s in cache_size:
    f.write(str(s)+' ')
f.write('\n')
for r in miss_rate.tolist():
    f.write(str(r)+' ')
f.write('\n')
f.close()

if args.showrank == 1:
    rank = cache.get_rank()
    age_rank = np.zeros(len(rank),)
    for i,r in enumerate(rank):
        age_rank[r-1] = i
    plt.title('age rank')
    plt.plot(age_rank.tolist())
    plt.show()
