# cosine_transform

[![Circle CI](https://circleci.com/gh/calvingiles/cosine_transform.svg?style=shield&circle-token=:circle-token=d5e42df9b83c0ceb1b6016a52b478c04abd984c9)](https://circleci.com/gh/calvingiles/cosine_transform)
[![Build Status](https://travis-ci.org/calvingiles/cosine_transform.svg?branch=master)](https://travis-ci.org/calvingiles/cosine_transform)
[![Coverage Status](http://coveralls.io/repos/calvingiles/cosine_transform/badge.svg?branch=master&service=github)](https://coveralls.io/github/calvingiles/cosine_transform?branch=master)

Module to transform a vector at a specified cosine distance from the first

## Install

```
pip install cosine_transform
```

## Usage

```python
>>> import cosine_transform
>>> import numpy
>>> from scipy.spatial.distance import cosine

>>> v = numpy.random.normal(size=1000)
>>> d = numpy.random.uniform(0, 2)
>>> u = cosine_transform.transform_to_distance(v, d)
>>> print(cosine(u, v), d)
(0.80948920647043165, 0.809489206470432)
>>> s = numpy.random.uniform(-1, 1)
>>> u = cosine_transform.transform_to_similarity(v, s)
>>> print(1 - cosine(u, v), s)
(0.94985416984017323, 0.9498541698401737)
```
