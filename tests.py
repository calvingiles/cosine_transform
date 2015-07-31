from __future__ import print_function, absolute_import, unicode_literals
import unittest

from cosinetransform import transform_to_similarity, transform_to_distance
import numpy

__author__ = 'calvin'


class TransformTestCase(unittest.TestCase):
    def setUp(self):
        vs = [[0, 1], [1, 0],
              [0, 0, 1]
              ]
        self.vs = [numpy.array(v) for v in vs]

    def test_vector_length_less_than_two(self):
        v = numpy.array([])
        self.assertRaises(ValueError, transform_to_similarity, *(v, 1))
        v = numpy.array([1])
        self.assertRaises(ValueError, transform_to_similarity, *(v, 1))

    def test_similarity_one(self):
        v = numpy.array([0, 1])
        u = transform_to_similarity(v, 1)
        self.assertTrue(numpy.all(u == v))

    def test_distance_zero(self):
        for v in self.vs:
            u = transform_to_distance(v, 0)
            self.assertTrue(numpy.all(u == v))


if __name__ == '__main__':
    unittest.main()
