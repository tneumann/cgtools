import numpy as np

__doc__ = """
Various functions for dealing with circular spaces and distributions
"""

twopi = np.pi * 2

def wrapped_mean(values, max_value):
    """ return the mean of values assuming a circular space that wraps at max_value """
    values = np.asanyarray(values)
    angles = (values*twopi) / max_value
    mean_angle = circular_mean(angles)
    return (mean_angle*max_value) / twopi

def circular_mean(angles):
    """ return the mean of values assuming a circular space 
        e.g. circular_mean([0.1, 2*pi-0.1]) == 0
    """
    angles = np.asanyarray(angles)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    if mean_angle < 0:
        mean_angle = twopi+mean_angle
    return mean_angle

def wrapped_distance(v1, v2, max_value = twopi):
    """ return the distance assuming distribution of v1 and v2
    where wrapping occurs at max_value """
    v1 = np.asanyarray(v1)
    v2 = np.asanyarray(v2)
    diff = np.abs(v1 - v2)
    return np.minimum(max_value - diff, diff)

