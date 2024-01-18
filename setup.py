from setuptools import setup, find_packages
import os
import codecs

KEYWORDS = ["python","astronomy","jwst","spectroscopy"]
INSTALL_REQUIRES = ["numpy","scipy","matplotlib","pandas","copy","astropy","jdaviz","reproject","photutils","specutils"]

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def read(*parts):
    with codecs.open(os.path.join(CURRENT_DIRECTORY, *parts), "rb", "utf-8") as f:
        return f.read()

if __name__ == "__main__":
    setup(
        name="teas", 
        version="0.0.2",
        author="Joelene Hales",
        author_email="<joelenehales@bell.net>",
        url="https://github.com/joelenehales/teas",
        description='Tools for the Extraction and Analysis of Spectra from JWST observations',
        long_description=read("README.rst"),
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        keywords=KEYWORDS,
        classifiers= [
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ]
    )