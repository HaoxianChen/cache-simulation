import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse
from cache import *


def mdp_repl_policy(hit_counter,evict_counter):
    value = np.zeros(MAX_AGE,dtype=float)
    new_value = np.zeros(MAX_AGE,dtype=float)
    delta= np.zeros(MAX_AGE,dtype=float)
    convergence = False
    theta = 0.01
    iteration_times = 4000
    logging.debug('\nValue Iteration Traces:\n')
    for i in range(iteration_times):
        logging.info('iteration: ' + str(i))
        for age in range(MAX_AGE-1):
            if sum(hit_counter[age:MAX_AGE]) > 0:
                hit_rate = float(hit_counter[age]) /float(sum(hit_counter[age:MAX_AGE])) 
            else:
                hit_rate = 0
            if sum(evict_counter[age:MAX_AGE]) > 0:
                evict_rate = float(evict_counter[age]) /float(sum(evict_counter[age:MAX_AGE])) 
            else:
                evict_rate = 0
            new_value[age] = hit_rate*(1+value[0])\
                + evict_rate*value[0] \
                + (1-hit_rate-evict_rate)*value[age+1]
            delta[age] = abs(new_value[age]-value[age])
            if new_value[age] > 0: 
                logging.info('age: ' + str(age) \
                    + ' value:' + str(new_value[age])) 
        logging.info('iteration: ' + str(i) + ' delta = ' + str(max(delta)))
        convergence =  max(delta) < theta
        value = np.copy(new_value)
        if convergence: return value

    return value

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
parser.add_argument("-r","--showrank", type=int, help='show rank function')
args = parser.parse_args()

small_array = range(0,16)
big_array = range(16,64)
iterate_times = 5000
steps = 1000

test_spec = str(args.s0) 
LOG_FILENAME = 'logs/trace-' + test_spec +'.log'
logging.basicConfig(filename=LOG_FILENAME, filemode='w+',level=logging.DEBUG)
logger = logging.getLogger(__name__)

FILENAME = 'logs/curve-' + test_spec +'.log'
f = open(FILENAME,'w')

f.write(str(iterate_times/steps) + '\n')

value = np.zeros(MAX_AGE,dtype=float)


for k in range(iterate_times/steps):
    cache = Cache(args.s0,value)
    k = weighted_choice([50,50])
    small_array_counter = 0
    big_array_counter = 0
    for i in range(steps):
        if i % 2 != 0:
            k = weighted_choice([149,2])
        else:
            k = weighted_choice([2,149])

        if k == 0:
            addr = small_array[small_array_counter % len(small_array)]
            small_array_counter += 1
        elif k == 1:
            addr = big_array[big_array_counter % len(big_array)]
            big_array_counter += 1
        cache.lookup(addr)

    value = np.copy(mdp_repl_policy(cache.get_hit_ages(),cache.get_evict_ages()))
    # log rank function
    for v in value:
        f.write(str(v)+' ')
    f.write('\n')

miss_rate = 1-cache.get_hit_rate()

# log hit age and eviction age distribution
for a in cache.get_hit_ages().tolist():
    f.write(str(a)+' ')
f.write('\n')
for a in cache.get_evict_ages().tolist():
    f.write(str(a)+' ')
f.write('\n')
