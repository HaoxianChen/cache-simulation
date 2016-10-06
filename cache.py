import numpy as np
import logging
import random

MAX_AGE = 150

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
