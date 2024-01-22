from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='filter_stations',
    version='0.6.1',
    packages=find_packages(),
    include_package_data=True,
    description='Making it easier to navigate and clean TAHMO weather station data for ML development',
    author='Austin Kaburia',
    author_email='kaburiaaustin1@gmail.com',
    url='https://github.com/kaburia/filter-stations',
    install_requires=[
        'pandas',
        'requests',
        'python-dateutil',
        'argparse',
        'haversine',
        'matplotlib',
        'numpy',
        'IPython',
        'folium',
        'datetime',
        'statsmodels',
        'tqdm',
        'geopandas',
        'matplotlib-scalebar',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'my-script=filter_stations.filter_stations:main'
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
