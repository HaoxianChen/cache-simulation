import sys
import compressedMDP as cmdp
import random
import numpy as np

def rollDice():
    p1 = random.randint(1,98) 
    p2 = random.randint(1,99-p1)
    p3 = 100 - p1 - p2 
    p1 = p1 / 100.
    p2 = p2 / 100.
    p3 = p3 / 100.
    p = [p1,p2,p3]
    
    d1 = random.randint(100,500) 
    d2 = random.randint(100,500) 
    d3 = random.randint(100,500)
    d = [d1,d2,d3]
    
    order = np.argsort(d)
    sorted_p = [p[i] for i in order]
    sorted_d = [d[i] for i in order]

    ed = round(np.dot(p,d))
    size = random.randint(1,ed)
    return sorted_p, sorted_d, size

if __name__ == '__main__':
    f = open('logs/monte_carlo.log','w')
    for i in range(1000):
        if i % 10 == 0:
            print "throwing dice " + str(i) 
        [p,d,s] = rollDice()
        ed = round(np.dot(p,d))
        if ed <= d[1]:  continue
        if d[0] == d[1] or d[1] == d[2]: continue

        m_analysis =  cmdp.opt_policy(p,d,s)[1]

        [h,e] = cmdp.parse_policy([0],p,d,s)[1:3]
        policy = cmdp.value_iteration(d,h,e,s,0.99)
        # policy = cmdp.policy_iteration(p,d,s)
        m = cmdp.parse_policy(policy,p,d,s)[0]

        if m-m_analysis > 1e-3:
            print 'error: ' + str([m,m_analysis,p,d,s])
            f.write(str([m,m_analysis,p,d,s])+'\n')
    
    f.close()
