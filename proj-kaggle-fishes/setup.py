from setuptools import setup
 
long_description = 'Looong description of your package, e.g. a README file'
 
setup(name='src', # name your package
      packages=['src', 'src.data', 'src.features', 'src.models', 'src.visualization'], # same name as above
      version='1.0.0', 
      description='Scripts for data analysis',
      long_description=long_description,
      url='http://www.kernix.com',
      author='Your Name',
      author_email='your.name@example.org',
      license='MIT')