#!/usr/bin/env python
from setuptools import setup

#TODO: make it so "tabulate.py" imports to main package, make adjustText an option module

setup(name='jacks_tools',
      version='0.1.2',
      description='Pre and post-processing tools for JACKS (felicityallen/JACKS).',
      author='John C. Thomas',
      author_email='jcthomas000@gmail.com',
      #url='',
      packages=['jacks_tools'],
      #modules=['front_end', 'tabulate']
      scripts=['front_end']
     )
