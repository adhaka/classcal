import os
import setuptools 
import codecs
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()
    
def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()
    

setup(
    name = 'rccpml', 
    version = '0.1.1', 
    description = 'Recalibrated and confident classification for probabilistic ML models',
    long_description = get_long_description(),
    long_description_context_type = 'text/markdown',
    author = 'Akash Kumar Dhaka',
    author_email = 'akash.dhaka@silo.ai',
    url = '',
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    python_requires='>=3.5',
    classifiers = ['Programming Language :: Python :: 3',
                    'Natural Language :: English',
                    'License :: OSI Approved :: MIT License',
                    'Intended Audience :: Science/Research/Developers',
                    'Operating System :: OS Independent',
                    'Development Status :: 4 - Beta',
                    'Topic :: Scientific/Engineering :: Mathematics'],
    keywords = 'robustness - recalibrated reliable probabilities machine learning classifiers',
    platforms = 'ALL', 
)


