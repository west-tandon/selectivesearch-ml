from setuptools import setup

setup(name='selectivesearch-ml',
      version='1.0',
      description='Research tools for Text Search and Information Retrieval - Machine Learning module',
      download_url='selectivesearch-ml',
      license='MIT',
      packages=['ossml'],
      scripts=['bin/ossml-impact', 'bin/ossml-cost'],
      install_requires=[
          'argparse',
          'scikit-learn',
          'scipy',
          'pandas',
          'fastparuqet'
      ])
