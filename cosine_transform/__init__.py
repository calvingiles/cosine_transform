from __future__ import print_function, absolute_import, unicode_literals
import numpy
from numpy import sin, cos, arccos, dot, isclose, count_nonzero
from scipy.spatial.distance import cosine, norm

__author__ = 'calvin'


def transform_to_similarity(v, s):
    if v.shape[0] < 2:
        raise ValueError('Cosine similarity cannot be < 1 for vectors with < 2 dimensions.')

    if count_nonzero(v) == 0:
        raise ValueError('All v elements are zero so solution undefined.')

    if (-1 <= s <= 1) == False:
        raise ValueError('Similarity is undefined for s not in range -1 <= s <= 1')

    if abs(s - 1) < 1e-8:  # s == 1
        return numpy.copy(v)
    if abs(s - -1) < 1e-8:  # s == -1
        return -v

    return _transform_to_similarity(v, s)


def _transform_to_similarity(v, s):
    v = v / norm(v)

    if s >= 0:
        u = numpy.copy(v)
    else:
        u = -v

    n = v.shape[0]
    m = numpy.abs(v).argmax()
    u[m] = 0

    if count_nonzero(v) == 1:
        u[(m+1)%n] = sin(arccos(s))

    A = dot(u, v)
    B = dot(u, u)

    a = v[m]**2 - s**2
    b = 2 * A * v[m]
    c = A**2 - B * s**2

    umx = -b / (2 * a)
    umy_inner = max(b**2 - 4*a*c, 0)  # To handle precision errors
    umy = numpy.sqrt(umy_inner) / (2 * a)

    um1 = umx + umy
    um2 = umx - umy

    if (
        abs((um1*v[m] + A) / numpy.sqrt(um1**2 + B) - s)
        < abs((um2*v[m] + A) / numpy.sqrt(um2**2 + B) - s)):
        u[m] = um1
    else:
        u[m] = um2

    u /= norm(u)

    return u


def transform_to_distance(v, d):
    return transform_to_similarity(v, 1 - d)