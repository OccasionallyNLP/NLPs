from multiprocessing import Pool
import os
import numpy as np
from collections import defaultdict
count = defaultdict(int)
def func(x:list):
    output = []
    for i in x:
        output.append(i*i)
    return output

def fun(x):
    print(os.getpid())
    return x**2

if __name__ == '__main__':
    temp = range(16)
    #t = np.array_split(list(temp),8)
    #a = list(map(lambda i:i.tolist(), t))
    pool = Pool(8)
    #d = pool.map(func, a)
    d = pool.map(fun, temp)
    pool.close()
    pool.join()
    print(count)