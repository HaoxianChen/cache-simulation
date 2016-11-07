import numpy as np
import logging
import random

MAX_AGE = 330
LOG_FILENAME = 'logs/cache.log'
logging.basicConfig(filename=LOG_FILENAME, filemode='w',level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Node:
    def __init__(self,addr):
        self.age = 0
        self.addr = addr
    def __str__(self):
        return 'addr: ' + str(self.addr) + ' age: ' + str(self.age)

class Cache:
    def __init__(self, size, value ):
        self.size = size
        self.data = []
        self.counter = -1
        self.value = value
        self.tags = []

        self.evict_candidate = None
        self.evict_threshold = max(value)

        # distributions
        self.hit_ages = np.zeros((MAX_AGE,), dtype=np.int)
        self.evict_ages = np.zeros((MAX_AGE,), dtype=np.int)

    def lookup(self,addr):
        # input: address
        # also need to update the age, hit, eviction distribution too
        self.counter += 1
        is_hit = False
        if self.size == 0: return

        threshold = self.evict_threshold
        self.evict_candidate = None
        # logger.debug(str(self.counter) + " access addr:" + str(addr))
        for i,node in enumerate(self.data):
            node.age += 1
            if node.age < MAX_AGE:
                value = self.value[node.age] 
            else:
                value = 0
            if value < self.evict_threshold and value < threshold:
                self.evict_candidate = i
                threshold = value
            if node.addr == addr:
                # hit
                is_hit = True
                # logger.debug('**hit** at age: ' + str(node.age) +' **' )
                if node.age < len(self.hit_ages):
                    self.hit_ages[node.age] += 1 
                # the age of hitted cnadicate is reset to 0
                node.age = 0
        if not is_hit:
            self.update(addr)
    
    def update(self,addr):
        if len(self.data) == self.size:
            # if cache is already full, evict a candidate
            victim_age = self.evict(addr)
            logger.info('evicted at age: ' + str(victim_age))
            # if victim_age == 0: self.log_data()
            if victim_age < len(self.evict_ages):
                self.evict_ages[victim_age] += 1

        elif len(self.data) < self.size:
            # compulsory miss
            new_node = Node(addr)
            self.data.append(new_node)

    def evict(self,addr):
        candidate = 0
        if self.evict_candidate != None and self.evict_candidate < self.size: 
            victim_age = self.data[self.evict_candidate].age
            candidate = self.evict_candidate
            # logger.info('found candidate at age: ' + str(victim_age))
        else:
            # randomly evict
            candidate = random.randrange(0,len(self.data),1)
            victim_age = self.data[candidate].age
            # logger.info('randomly evict at age: ' + str(victim_age))
            self.log_data()
        self.data[candidate].addr = addr
        self.data[candidate].age = 0
        assert len(self.data) == self.size
        return victim_age

    def get_hit_ages(self):
        return self.hit_ages

    def get_evict_ages(self):
        return self.evict_ages

    def get_hit_rate(self):
        hit_times = sum(self.hit_ages)
        # To get the long term hit rate, just minus the compulsory misses
        return float(hit_times)/float(self.counter-self.size)

    def log_data(self):
        logger.info('cache data at ' + str(self.counter) + ' access: ')
        for n in self.data:
            logger.info(str(n))

    def get_data(self):
        return self.data
