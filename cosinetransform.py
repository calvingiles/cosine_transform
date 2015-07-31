from __future__ import print_function, absolute_import, unicode_literals
import numpy
from numpy import sin, arccos
from scipy.spatial.distance import cosine, norm

__author__ = 'calvin'


def transform_to_similarity(v, s):
    if v.shape[0] < 2:
        raise ValueError('Cosine similarity cannot be < 1 for vectors with < 2 dimensions.')

    non_zero_v = (v != 0).sum()
    if non_zero_v == 0:
        raise ValueError('All v elements are zero so solution undefined.')

    if s == 1:
        return numpy.copy(v)

    v = v / norm(v)

    if s >= 0:
        u = numpy.copy(v)
    else:
        u = -v

    n = v.shape[0]
    m = numpy.abs(v).argmax()
    u[m] = 0

    if non_zero_v == 1:
        u[(m+1)%n] = sin(arccos(s))

    A = norm(u, v)
    B = norm(u, u)

    a = v[m]**2 - s**2
    b = 2 * A * v[m]
    c = A**2 - B * s**2

    umx = -b / (2 * a)
    umy = numpy.sqrt(b**2 - 4*a*c) / (2 * a)

    um1 = umx + umy
    um2 = umx - umy

    u[m] = um1
    e1 = abs(cosine(u, v) - s)
    u[m] = um2
    e2 = abs(cosine(u, v) - s)
    if e1 < e2:
        u[m] = um1

    return u


def transform_to_distance(v, d):
    return transform_to_similarity(v, 1 - d)