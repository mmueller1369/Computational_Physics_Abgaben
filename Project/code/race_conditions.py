"""prange issue"""
import numpy as np
from numba import njit, prange
from time import time

@njit(parallel=True)
def no_race_condition_simple():
    total = 0
    for i in prange(100000):  # Numba recognizes this as a sum reduction
        total += 1
    return total  # Should be exactly 100000

@njit(parallel=True)
def sliced_array_no_race():
    """
    each thread runs one part of the loop. Each thread only access array indices in their range
    -> not possible to access index from other thread.
    """
    n= 1000
    a = np.zeros(1000)
    for i in prange(n):
        a[i] += 1  
    return max(a), a.sum()

@njit(parallel=True)
def race_condition_prange():
    total = np.zeros(1)  # Shared array
    n = 100000
    for i in prange(n):
        total[0] += 1  # This will be corrupted due to race conditions
        # multiple threads access the same value
    return total[0] , n/total[0]

@njit(parallel = True)
def race_condition_modulo():
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    a = np.zeros(4)
    n=  len(x)
    for i in prange(n):
        a[i%4] += x[i]  # This will lead to race conditions
    b = np.zeros(4)
    for i in range(n):
        b[i%4] += x[i]
    return a, b

@njit(parallel = True)
def race_condition_integer_division():
    """
    Threads accessing indices from other Threads, because using // 
    I would expect: 
    """
    n = 10
    a = np.zeros(n)
    for i in prange(n):    
        a[i//3] += 1 
    b = np.zeros(n)
    for i in range(n):
        b[i//3] += 1
    return a, b

def double_range():
    N = 100
    x = np.zeros(N)
    for i in range(N-1):
        for j in range(i+1, N):
            x[i] -= 1
            x[j] += 1
    return x

@njit(parallel= True)
def double_prange():
    N = 100
    x = np.zeros(N)
    for i in prange(N-1):
        for j in prange(i+1, N):
            x[i] -=1
            x[j] += 1
    return x


if __name__ == "__main__":
#    #example 1 
#    print("Expect: 10000 result: ", no_race_condition_simple())
#    print("Expect: (1,1000) result: ",sliced_array_no_race())
#    # example 2
#    print("Expect (100000, _) result: ", race_condition_prange())
#    res = race_condition_modulo()
#    print(f"Expect: {res[1]}, result: {res[0]}")
#
#    res = race_condition_integer_division()
#    print(f"Expect: {res[1]}, result: {res[0]}")

    print("range: ", double_range())
    print("prange: ", double_prange())#.parallel_diagnostics(level = 4))