from distutils.core import setup

setup(
    name='cosine_transform',
    version='0.0.2',
    packages=['cosine_transform'],
    url='https://github.com/calvingiles/cosine_transform',
    download_url='https://github.com/calvingiles/cosine_transform/tarball/0.0.2',
    license='LICENCE.txt',
    author='calvin',
    author_email='calvin.giles@gmail.com',
    description='Transform a vector to a new vector a given cosine distance away.',
    install_requires=['numpy>=1.8.0', 'scipy>=0.13.2'],
)
