#!/usr/bin/python

import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import random

class LinkedList:
    # nodes are (data, prev, next)

    def __init__(self):
        self.head = ['Head', None, None]
        self.tail = ['Tail', None, None]
        self.head[2] = self.tail
        self.tail[1] = self.head
        self.length = 0

    def push(self, data):
        assert data != None
        # print 'Insert ', data

        # node = { .data = data, .prev = tail.prev, .next = tail }
        node = [data, self.tail[1], self.tail]

        # tail.prev.next = node
        self.tail[1][2] = node

        # tail.prev = node
        self.tail[1] = node

        self.length += 1
        return node

    def remove(self, node):
        # print 'Remove ', node[0]

        # node.prev.next = node.next
        node[1][2] = node[2]

        # node.next.prev = node.prev
        node[2][1] = node[1]

        self.length -= 1

    def pop(self):
        # data = self.tail.prev.data
        data = self.tail[1][0]

        self.remove(self.tail[1])
        # print 'Return ', data
        return data

    def debugPrint(self):
        p = self.head
        while p != None:
            # print p.data
            print p[0],

            # p = p.next
            p = p[2]

class MRUCache:
    def __init__(self, size, allowBypass = False):
        self.mru = LinkedList()
        self.tags = []
        self.accesses = 0
        self.hits = 0
        self.evictions = 0
        self.fills = 0
        self.bypasses = 0
        self.allowBypass = allowBypass
        self.size = size

    def access(self, addr):
        if len(self.tags) <= addr:
            self.tags.extend([None] * ((addr+1) - len(self.tags)))
            assert (len(self.tags)) == addr+1

        if self.tags[addr] == None:

            if self.mru.length < self.size:
                self.fills += 1
            elif self.allowBypass:
                self.bypasses += 1
                self.accesses += 1
                return
            else:
                # evict mru
                victim = self.mru.pop()
                self.tags[victim] = None
                self.evictions += 1

        else:
            self.mru.remove(self.tags[addr])
            self.hits += 1

        self.tags[addr] = self.mru.push(addr)

        self.accesses += 1

        # self.mru.debugPrint()
        # raw_input()

    def go(self, trace):
        print "Simulating %d " % self.size, 
        for i in range(len(trace.trace)):
            if i % (len(trace.trace)/50) == 0: 
                sys.stdout.write('.')
                sys.stdout.flush()
            self.access(trace.trace[i])
        print

def weightedChoice(weights):
    total = sum(w for w,c in weights)
    r = random.uniform(0, total)
    upto = 0
    for w, c in weights:
        upto += w
        if r <= upto:
            return c
    assert False, "Unreachable"

class GrowingList(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([0]*(index + 1 - len(self)))
        list.__setitem__(self, index, value)
    def __getitem__(self, index):
        if index >= len(self):
            self.extend([0]*(index + 1 - len(self)))
        return list.__getitem__(self, index)
    def add(self, that):
        for i in range(len(that)):
            self[i] += that[i]

class GrowingMatrix(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([GrowingList() for i in range(index+1-len(self))])
        list.__setitem__(self, index, value)
    def __getitem__(self, index):
        if index >= len(self):
            self.extend([GrowingList() for i in range(index+1-len(self))])
        return list.__getitem__(self, index)
    def add(self, that):
        for i in range(len(that)):
            self[i].add(that[i])

class Trace:
    def __init__(self):
        self.trace = []
        self.rds = []
        self.rdDist = GrowingList()
        # self.prevRdDist = GrowingMatrix()
        self.index = 0

    def histogram(self):
        print "Histogram ",
        maxAddress = max(self.trace)
        prevUse = [ None for a in range(maxAddress+1) ]
        prevPrevUse = [ None for a in range(maxAddress+1) ]
        self.rds = [None for i in self.trace]
        for i in range(len(self.trace)):
            if i % (len(self.trace)/50) == 0: 
                sys.stdout.write('.')
                sys.stdout.flush()
            a = self.trace[i]
            if prevUse[a] != None:
                rd = i - prevUse[a]
                assert(self.rds[i-rd] == None)
                self.rds[i-rd] = rd
                self.rdDist[rd] += 1
                # if prevPrevUse[a] != None:
                #     prevRd = prevUse[a] - prevPrevUse[a]
                #     self.prevRdDist[prevRd][rd] += 1
                # prevPrevUse[a] = prevUse[a]
            prevUse[a] = i
        print
        self.rdDist = np.array(self.rdDist)

class TraceScan(Trace):
    def __init__(self, probabilities, sizes):
        Trace.__init__(self)
        self.probs = zip(probabilities, range(len(probabilities)))
        self.sizes = sizes
        self.counters = [ 0 for x in probabilities ]
        self.addrOffsets = 1 + np.cumsum(sizes)

    def nextAddr(self):
        # select which loop to scan over.
        sel = weightedChoice(self.probs)

        # scan
        self.counters[sel] += 1
        self.counters[sel] %= self.sizes[sel]

        return self.addrOffsets[sel] + self.counters[sel]

    def generate(self, n):
        self.trace = []
        for i in range(n):
            self.trace.append(self.nextAddr())
        self.histogram()

# Try to generate a trace matching a target distribution as closely as
# possible, with random draws of reuse distance at each reuse.
class TraceDistribution(Trace):
    def __init__(self, dist):
        Trace.__init__(self)
        self.dist = dist

    def generate(self, size):
        addr = 1
        error = 0
        filled = [0]

        def fill():
            filled[0] += 1
            if filled[0] % (size/50) == 0: 
                sys.stdout.write('.')
                sys.stdout.flush()

        self.trace = [0] * size

        print "Generating",
        while filled[0] < size:
            start = 0
            while self.trace[start] != 0:
                start += 1
                assert(start < size)

            self.trace[start] = addr
            fill()

            cur = start
            while cur < size:
                # find where this should be reused next. place as close as possible
                rd = weightedChoice(self.dist)
                if rd < 0:
                    # dead line
                    continue
        
                j = 0
                while True:
                    # place as close as possible
                    dest = cur + rd - j
                    if cur < dest and dest < size and self.trace[dest] == 0:
                        self.trace[dest] = addr
                        fill()
                        break

                    dest = cur + rd + j
                    if cur < dest and dest < size and self.trace[dest] == 0:
                        self.trace[dest] = addr
                        fill()
                        break

                    # no spot available
                    if cur + rd - j < cur and cur + rd + j > size:
                        break

                    # try next spot
                    else:
                        j += 1

                cur = dest
                error += j

            addr += 1
        print 

        self.histogram()
        self.rdDist = self.rdDist[ : 2*max(c for w,c in self.dist) ]

        print "Distribution:", self.dist
        print "Addresses up to:", addr-1
        print "Average error:", 1. * error / size
        print "Histogram weight:", 1. * np.sum(self.rdDist) / len(self.trace)

if __name__ == '__main__':

    idealRdDist = [(0.25,64),(.25,192),(.5,512)]

# trace = TraceScan([p for p,d in idealRdDist], [int(p*d+0.5) for p,d in idealRdDist])
# trace.generate(100000)

    trace = TraceDistribution(idealRdDist)
    trace.generate(100000)

    plt.figure()
    def idealRdDistNonSparse():
        cump = 0.
        i = 0
        rdd = np.zeros_like(trace.rdDist)
        for d in range(len(rdd)):
            while i < len(idealRdDist) and idealRdDist[i][1] <= d:
                cump += idealRdDist[i][0]
                i += 1
            rdd[d] = cump * len(trace.trace)
        return rdd
    plt.plot(idealRdDistNonSparse(), label='Ideal RD dist')
    plt.plot(np.cumsum(trace.rdDist), label='Actual RD dist')
    plt.legend(loc='best')
    plt.show()

    sizes = range(1, 320, 8)
    hits = np.zeros_like(sizes)

    for i in range(len(sizes)):
        cache = MRUCache(sizes[i], allowBypass = False)
        cache.go(trace)

        hits[i] = cache.hits

        print 'Accesses %s, Hits %s, Evictions %s, Fills %s' % (cache.accesses, cache.hits, cache.evictions, cache.fills)

    def modelNathan(size):
        return size / np.sum( p * d for p,d in idealRdDist )

    def modelHaoxian(size):
        return np.sum( p * size / d for p,d in idealRdDist )

    plt.figure()
    plt.plot(sizes, 1. * hits / len(trace.trace), marker='.', label='Simulation')

    sizes = np.arange(max(sizes))
    plt.plot(sizes, modelNathan(sizes), label=r'$S/E(D)$')
    plt.plot(sizes, modelHaoxian(sizes), label=r'$\sum_i p_i S / d_i$')

    plt.xlim(0,sizes[-1])
    plt.ylim(0,1)
    plt.legend(loc='best')
    plt.show()
