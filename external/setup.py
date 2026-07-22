from setuptools import setup, find_packages

setup(
    name="bempp-cl",
    version="0.3.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
    ],
)