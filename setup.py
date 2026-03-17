from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# requirements are in requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='WeatherMashariki',
    version='0.8.0',
    packages=find_packages(),
    include_package_data=True,
    description='A secure, unified Python interface for African climate data, integrating TAHMO station data and gridded datasets (IMERG, CHIRPS, ERA5, TAMSAT), and medium-to-seasonal weather models',
    author='Austin Kaburia',
    author_email='kaburiaaustin1@gmail.com',
    url='https://github.com/kaburia/filter-stations',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'my-script=filter_stations.filter_stations:main'
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
