import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

setup(
    name='ml',
    ext_modules=cythonize([
        Extension(
            "fast_eval",
            sources=["metrics/fast_evaluator.pyx"],
            include_dirs=[np.get_include()]
        ), ],
        language_level="3"
    ),
    install_requires=["numpy"]
)
