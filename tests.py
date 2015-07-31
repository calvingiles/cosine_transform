from __future__ import print_function, absolute_import, unicode_literals
import unittest

import cosinetransform as ct
import numpy
from scipy.spatial.distance import norm, cosine

__author__ = 'calvin'


class VariableVTransformTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = numpy.random.uniform(-1, 1, 10)
        self.oob_ss = numpy.random.uniform(1, numpy.inf, 1000)
        self.oob_dsp = numpy.random.uniform(2, numpy.inf, 500)
        self.oob_dsm = numpy.random.uniform(-numpy.inf, 0, 500)
        vs = []
        # axis vectors and 45 degree vectors
        n_max = 10
        for n in range(2, n_max+1):
            for i in range(n):
                for j in range(n):
                    v = [int((i==idx) or (j==idx)) for idx in range(n)]
                    vs.append(numpy.array(v))

        n_repeats = 100
        for _ in range(n_repeats):
            vs.append(numpy.random.uniform(-1, 1, numpy.random.randint(2, 10000)))

        self.vs = vs

    def test_vector_length_zero(self):
        v = numpy.array([])
        s = 1
        self.assertRaises(ValueError, ct.transform_to_similarity, *(v, s))

    def test_vector_length_one(self):
        v = numpy.array([1])
        s = 1
        self.assertRaises(ValueError, ct.transform_to_similarity, *(v, s))

    def test_similarity_one(self):
        for v in self.vs:
            u = ct.transform_to_similarity(v, 1)
            self.assertTrue(numpy.allclose(u, v))

    def test_similarity_minus_ones(self):
        for v in self.vs:
            u = ct.transform_to_similarity(v, -1)
            self.assertTrue(numpy.allclose(u, -v))

    def test_similarity_out_of_bounds(self):
        v = numpy.array([0, 1])
        for s in self.oob_ss:
            if numpy.isclose(s, 1) or numpy.isclose(s, -1):
                continue
            self.assertRaises(ValueError, ct.transform_to_similarity, *(v, s))
            self.assertRaises(ValueError, ct.transform_to_similarity, *(v, -s))

    def test_distance_zero(self):
        for v in self.vs:
            u = ct.transform_to_distance(v, 0)
            self.assertTrue(numpy.allclose(u, v))

    def test_distance_two(self):
        for v in self.vs:
            u = ct.transform_to_distance(v, 2)
            self.assertTrue(numpy.allclose(u, -v))

    def test_distance_out_of_bounds(self):
        v = numpy.array([0, 1])
        for d in self.oob_dsp:
            if numpy.isclose(d, 2):
                continue
            self.assertRaises(ValueError, ct.transform_to_distance, *(v, d))
        for d in self.oob_dsm:
            if numpy.isclose(d, 0):
                continue
            self.assertRaises(ValueError, ct.transform_to_distance, *(v, d))

    def test_similarity(self):
        for v in self.vs:
            for s in self.ss:
                u = ct.transform_to_similarity(v, s)
                s2 = 1 - cosine(u, v)
                self.assertAlmostEquals(s, s2)


if __name__ == '__main__':
    unittest.main()
