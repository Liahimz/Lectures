from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

setup(
    ext_modules=cythonize(os.path.abspath(os.path.dirname(__file__)) + "/model_inference.pyx"),
    include_dirs=[numpy.get_include()]
)
