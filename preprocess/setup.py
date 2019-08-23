#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-19 17:28:27
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules=cythonize(Extension(
    '_max_clustering_cython',
    sources=['_max_clustering_cython.pyx'],
    language='c++',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
