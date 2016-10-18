import numpy as np
import logging
import random

MAX_AGE = 150
LOG_FILENAME = 'logs/cache'
logging.basicConfig(filename=LOG_FILENAME, filemode='w',level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, size, value ):
        self.size = size
        self.data = []
        self.counter = -1
        self.value = value

        # distributions
        self.hit_ages = np.zeros((MAX_AGE,), dtype=np.int)
        self.evict_ages = np.zeros((MAX_AGE,), dtype=np.int)

    def lookup(self,addr):
        # input: address
        # output: data
        # also need to update the age, hit, eviction distribution too
        self.age()
        self.counter += 1
        for n in self.data:
            if n['addr'] == addr:
                if n['age'] < len(self.hit_ages):
                    self.hit_ages[n['age']] += 1 
                # the age of hitted cnadicate is reset to 0
                n['age'] = 0
                return n['data']
        # if address not found, update cache
        return self.update(addr)

    def age(self):
        # the hitted or evicted candidate's age is being reset to zeor, and all
        # the rest candidates ages by +1.
        for n in self.data:
            n['age'] += 1
        return
    
    def update(self,addr):
        if len(self.data) == self.size:
            # if cache is already full, evict a candidate
            victim_age = self.evict()
            logger.info('evicted candidate at age: ' + str(victim_age))
            if victim_age < len(self.evict_ages):
                self.evict_ages[victim_age] += 1
        new_node = {}
        new_node['addr'] = addr
        new_node['data'] = random.randint(0,100)
        new_node['age'] = 0
        self.data.append(new_node)
        return new_node['data']

    def evict(self):
        # given the values of each age, evict the age of lowest value
        is_evicted = False
        local_values = list(self.value)
        while not is_evicted:
            victim_age = random.choice([i for i,x in enumerate(local_values) if x == min(local_values)])
            if victim_age == 0:
                # just don't cache the accessing data
                is_evicted = True
            for n in self.data:
                if n['age'] == victim_age:
                    self.data.remove(n)
                    is_evicted = True
                elif n['age'] > MAX_AGE:
                    self.data.remove(n)
                    is_evicted = True
                    victim_age =  n['age']
            # if candidate not found, search the next available candidate
            local_values[victim_age] = max(local_values) + 1
        return victim_age

    def get_hit_ages(self):
        return self.hit_ages

    def get_evict_ages(self):
        return self.evict_ages

    def get_hit_rate(self):
        hit_times = sum(self.hit_ages)
        return float(hit_times)/float(self.counter)

    def log_data(self):
        logger.info('cache data at ' + str(self.counter) + ' access: ')
        for n in self.data:
            logger.info(str(n))

    def get_data(self):
        return self.data
