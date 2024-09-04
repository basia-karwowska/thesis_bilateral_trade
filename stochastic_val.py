import numpy as np

'''
def stochastic_valuations(dist1, dist2, *args, **kwargs):
    """
    Generates random samples from two given distributions.

    Parameters:
    - dist1: a callable that represents the first distribution function.
    - dist2: a callable that represents the second distribution function.
    - *args1: positional arguments for the first distribution function.
    - **kwargs1: keyword arguments for the first distribution function.
    - *args2: positional arguments for the second distribution function.
    - **kwargs2: keyword arguments for the second distribution function.
    
    Returns:
    - A tuple containing one random sample from each distribution.
    """
    sample1 = dist1(*args, **kwargs)
    sample2 = dist2(*args, **kwargs)
    return sample1, sample2

'''

# We reasonably assume that seller's valuations are slightly right-skewed
# and buyer's slightly left-skewed and both in the interval [0, 1]
def stochastic_valuations(a_b, b_b, a_s, b_s):
    return np.random.beta(a=a_b, b=b_b), np.random.beta(a=a_s, b=b_s)
