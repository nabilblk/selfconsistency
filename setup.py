#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    reqs = list(filter(lambda x : x.startswith('#'),f.readlines()))
with open('README.md', 'r') as f:
    readme = f.read()
setup(
    name='selfconsistency',
    version='1.4',
    packages=find_packages(exclude=['tests*']),
    license='Apache 2.0',
    description='Image Splice Detection via Learned Self-Consistency',
    long_description=readme,
    install_requires=reqs,
    url='https://github.com/AXATechLab/selfconsistency/'
)
