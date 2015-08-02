from distutils.core import setup

version = '0.0.3'

setup(
    name='cosine_transform',
    version=version,
    packages=['cosine_transform'],
    url='https://github.com/calvingiles/cosine_transform',
    download_url='https://github.com/calvingiles/cosine_transform/tarball/{}'.format(version),
    license='LICENCE.txt',
    author='calvin',
    author_email='calvin.giles@gmail.com',
    description='Transform a vector to a new vector a given cosine distance away.',
    install_requires=['numpy>=1.8.0', 'scipy>=0.13.2'],
    tests_requires=['coverage>=3.7.1', 'nosetests'],
)
