from setuptools import setup, find_packages

setup(
    name='weslab',
    version='0.1.0',
    packages=find_packages(),
    packages=find_packages(include=["weslab", "weslab.*"]),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'numpy',
        'xarray',
        'matplotlib',
        'scipy',
        'pyproj',
        'openmdao',
        'py_wake', 
        'utm',
        'openpyxl',
        'geopandas',
        'shapely',
        'geojson'
    ],
    author='Rafael Valotta Rodrigues',
    author_email='r.valottarodrigues@umb.edu',
    description='Optimizer for offshore systems.',
    url='https://github.com/rafaelvalotta/weslab',
    classifiers=[
        'Programming Language :: Python :: 3.11.11',
    ],
)
