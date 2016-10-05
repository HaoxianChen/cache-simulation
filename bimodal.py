import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime
import logging
import argparse


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

class Cache:
    def __init__(self,size = 32):
        self.size = size
        self.data = []
        self.counter = -1

        # distributions
        self.hit_ages = np.zeros((MAX_AGE,), dtype=np.int)
        self.evict_ages = np.zeros((MAX_AGE,), dtype=np.int)

    def lookup(self,addr):
        # input: address
        # output: data
        # also need to update the age, hit, eviction distribution too
        self.age()
        self.counter += 1
#        self.log_data()
        for n in self.data:
            if n['addr'] == addr:
#                logging.info('hit addr: ' + str(addr) + ' age: ' +
#                        str(n['age']))
                # update the hit age distribution
                if n['age'] < len(self.hit_ages):
                    self.hit_ages[n['age']] += 1 
                # the age of hitted cnadicate is reset to 0
                n['age'] = 0
                return n['data']
#        logging.info('miss addr: ' + str(addr))
        # if address not found, update cache
        return self.update(addr)

    def age(self):
        # the hitted or evicted candidate's age is being reset to zeor, and all
        # the rest candidates ages by +1.
        for n in self.data:
            n['age'] += 1
        return
    
    def repl_policy(self):
        # return a ranked list of eviction age
        rank = range(33,100)+range(1,33)
        return rank

    def update(self,addr):
        if len(self.data) == self.size:
            # if cache is already full, evict a candidate
            victim_age = self.evict()
            logging.info('evicted candidate at age: ' + str(victim_age))
            if victim_age < len(self.evict_ages):
                self.evict_ages[victim_age] += 1
        new_node = {}
        new_node['addr'] = addr
        new_node['data'] = random.randint(0,100)
        new_node['age'] = 0
        self.data.append(new_node)
        return new_node['data']

    def evict(self):
        rank = self.repl_policy()
        for victim_age in rank:
            for n in self.data:
                if n['age'] == victim_age:
                    self.data.remove(n)
                    return victim_age
                elif n['age'] > MAX_AGE:
                    self.data.remove(n)
                    logging.info('evicted candidate at age' + str(n['age']))
                    return n['age']
        # if no desired age candidate found, randomly evict a candidate
        logging.info(str(self.size)+'-have to randomly kick an candidate')
        victim_index = random.randint(0,len(self.data)-1)
        victim_age = self.data[victim_index]['age']
        del self.data[victim_index]
        return victim_age

    def get_hit_ages(self):
#        return np.divide(self.hit_ages,sum(self.hit_ages),dtype=float)
        return self.hit_ages
            
    def get_evict_ages(self):
        return self.evict_ages

    def get_hit_rate(self):
        hit_times = sum(self.hit_ages)
        return float(hit_times)/float(self.counter)

    def log_data(self):
        logging.info('cache data at ' + str(self.counter) + ' access: ')
        for n in self.data:
            logging.info(str(n))

    def get_data(self):
        return self.data

    def get_rank(self):
        return self.repl_policy()

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
