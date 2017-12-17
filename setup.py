from setuptools import setup

LONG_DESCRIPTION = """
**tinder** provides extra layers and helpers for Pytorch.
"""

setup(name='tinder',
      version='0.1',
      description='Pytorch helpers and utils',
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering'
      ],
      url='http://github.com/elbaro/tinder',
      author='elbaro',
      author_email='elbaro@users.noreply.github.com',
      license='MIT',
      packages=['tinder'],
      keywords=['tinder','pytorch','torch'],
      zip_safe=False)

