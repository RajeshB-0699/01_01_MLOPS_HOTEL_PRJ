from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = "MLOps Project 1",
    author = "Rajesh B",
    version = "1.0",
    packages = find_packages(),
    install_requires = requirements
)