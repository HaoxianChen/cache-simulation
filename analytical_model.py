import numpy as np
import matplotlib.pyplot as plt

def analysis_modal(p,d,h,e,allow_plot=False):
    assert abs(sum(p) - 1. ) < 1e-4
    assert len(p) == len(d)
    n = int(max(d) * 1.2)

    rdd = np.zeros(n)
    for i in range(len(d)):
        rdd[d[i]] = p[i]

    a = np.zeros_like(d,dtype=float)
    b = np.zeros_like(d,dtype=float)

    ed = sum([p[i]*d[i] for i in range(len(p))])

    cumevents = np.cumsum((h+e)[::-1])[::-1]
    cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

    a[0] = 1. / d[0]
    for i in range(1,len(d)):
        left_over = 1. - (h[d[i-1]] - e[d[i-1]]) / cumevents[d[i-1]]
        hit = h[d[i-1]] / cumevents[d[i-1]]

        delta_d = d[i]-d[i-1]

        a[i] = a[i-1] * left_over 
        a[i] /= 1 + a[i-1] * left_over * delta_d + a[i-1]

        #b[i] = a[i] * hit / left_over + b[i-1]
        b[i] = a[i-1] * hit + b[i-1]
        b[i] /= 1 + a[i-1] * left_over * delta_d + a[i-1]
    
    slope = a[-1] + b[-1]

    values = np.arange(n) * slope
    values[max(d)+1:] = 0
    for i in range(1,len(d)):
        left_over = 1. - (h[d[i-1]] - e[d[i-1]]) / cumevents[d[i-1]]
        hit = h[d[i-1]] / cumevents[d[i-1]]

        next_value = values[d[i-1]] + slope - hit
        next_value /= left_over
    
        gap = values[d[i-1]] - next_value

        values[d[i-1]+1:] -= gap

    if allow_plot:
        plt.plot(values)
        plt.show()

    return values

if __name__ == '__main__':
    p = [1,0,0]
    d = [40,5,200]
    
    rdd =np.zeros(max(d))
    rdd[4] = 1
    plt.plot(rdd)
    plt.xlabel('age')
    plt.ylabel('probability')
    plt.title('reuse distance distribution')
    plt.show()


    print "input parameters:"
    print "reuse distances " + str(d)
    print "probabilities " + str(p)
    print "\ncritical outputs"
    values = analysis_modal(p,d,True)
    print "v[0] = %.3f" %values[0]
    print "v[1] = %.3f" %values[1]
    for i in range(len(d)):
        print "v[%d] = %.3f" %(d[i]+1,values[d[i]+1])
