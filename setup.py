#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='selfconsistency',
      version='1.0',
      description='Image Splice Detection via Learned Self-Consistency',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE.txt',
    )