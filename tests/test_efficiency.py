import numpy as np

def exp_star(a,b):
    """
    195 ns ± 2.93 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    371 ns ± 4.73 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    211 ns ± 2.02 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    xy
    1.17 µs ± 16.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    1.18 µs ± 12.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


    2.09 µs ± 13.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    """
    x = a**b

def exp_np(a,b):
    """
    1.92 µs ± 14.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    2.1 µs ± 9.85 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    1.99 µs ± 9.85 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    xy
    1.21 µs ± 11.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    1.22 µs ± 14.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    2.18 µs ± 52 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    """
    x = np.power(a,b)

def exp_np_float(a,b):
    """
    1.59 µs ± 13.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    1.71 µs ± 14.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    1.58 µs ± 18.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    xy
    888 ns ± 11 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    896 ns ± 9.69 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    1.81 µs ± 6.96 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    x = np.float_power(a,b)

a = np.power(10., 3*(2*np.random.rand()-1))
b = np.power(10., 3*(2*np.random.rand()-1))
%timeit exp_star(a,b)
%timeit exp_np(a,b)
%timeit exp_np_float(a,b)

x = 3*np.random.rand(10)
y = 3*(2*np.random.rand(10)-1)
print("exp_star")
%timeit exp_star(x,y)
print('exp_np')
%timeit exp_np(x,y)
print('exp_np_float')
%timeit exp_np_float(x,y)


x = np.power(10., 3*(2*np.random.rand(10)-1))
y = np.power(10., 3*(2*np.random.rand(10)-1))
print("exp_star")
%timeit exp_star(x,y)
print('exp_np')
%timeit exp_np(x,y)
print('exp_np_float')
%timeit exp_np_float(x,y)