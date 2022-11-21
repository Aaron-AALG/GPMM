from setuptools import find_packages, setup

with open("README.rst", "r") as readme:
    long_description = readme.read()

setup(
    name = 'GPMM',
    packages = find_packages(include=['GPMM']),
    version = '0.1.2',
    author = 'Aaron Lopez-Garcia',
    author_email='aaron.lopez@uv.es',
    description = 'Collection of generalized p-mean models with classic, fuzzy and unweighted approach',
    long_description=long_description,
    license = 'MIT',
    url='https://github.com/Aaron-AALG/GPMM',
    download_url = 'https://github.com/Aaron-AALG/GPMM/releases/tag/GPMM_0.1.2',
    install_requires=['numpy >= 1.19',
                      'scipy >= 1.6.3'],
    classifiers=["Programming Language :: Python :: 3.8",
			     "License :: OSI Approved :: MIT License"],
)
