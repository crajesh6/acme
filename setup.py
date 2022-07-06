"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

setup(
    name="acme",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.6, <4",

)
