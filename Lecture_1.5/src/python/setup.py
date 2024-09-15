from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("/home/michael/Desktop/Lectures/Lecture_1.5/src/python/model_inference.pyx"),
    include_dirs=[numpy.get_include()]
)
