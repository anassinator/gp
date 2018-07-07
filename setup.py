#!/usr/bin/env python
"""Gaussian Process setup."""

import os
from setuptools import setup

__version__ = "0.1.0"


def read(fname):
    """Reads a file's contents as a string.

    Args:
        fname: Filename.

    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


BASE_URL = "https://github.com/anassinator/gp"
INSTALL_REQUIRES = [
    "numpy==1.14.2",
    "six==1.11.0",
    "torch==0.4.0",
]

setup(
    name="gp",
    version=__version__,
    description="Gaussian Process implementation for PyTorch",
    long_description=read("README.rst"),
    author="Anass Al",
    author_email="dev@anassinator.com",
    license="MIT",
    url=BASE_URL,
    download_url="{}/tarball/{}".format(BASE_URL, __version__),
    packages=["gp"],
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
