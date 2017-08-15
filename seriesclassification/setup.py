from distutils.core import setup
import os
from os.path import join
import warnings

setup(
    name='seriesclassification',
    version='1.0',
    packages=['test', 'tools', 'utils', 'shapelet', 'classifier', 'classifier.cnn', 'clustering'],
    url='',
    license='',
    author='huangfanling',
    author_email='',
    description='time series mining package'
)


def configuration(parent_name='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration(package_name='tsmining',
                           parent_name=parent_name,
                           top_path=top_path)

    # submodules with build utilities

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('classifier')
    config.add_subpackage('shapelet')
    config.add_subpackage('tools')
    config.add_subpackage('utils')

    # submodules which have their own setup.py

    # add the test directory
    config.add_subpackage('test')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())



