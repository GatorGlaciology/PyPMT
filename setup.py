from setuptools import setup

version = '1.0.6'

classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Information Analysis',
]

dependencies = [
    'Cartopy>=0.22.0',
    'geopy>=2.3.0',
    'matplotlib>=3.5.2',
    'numpy>=2.0.1',
    'pyproj>=3.5.0',
    'scipy>=1.14.0',
    'Shapely>=2.0.5,
]

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'PyPMT',
    version = version,
    description = 'Python package for analysis and visualization of polar datasets.',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    author = 'Blake Wood',
    author_email = 'blakewood1@ufl.edu',
    url = 'https://github.com/GatorGlaciology/PyPMT',
    license = 'MIT',
    classifiers = classifiers,
    py_modules = ['PyPMT'],
    python_requires = '>=3',
    install_requires = dependencies,
)