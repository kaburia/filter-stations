from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# requirements are in requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='filter_stations',
    version='0.7.1',
    packages=find_packages(),
    include_package_data=True,
    description='Making it easier to navigate and clean TAHMO weather station data and creating a way to access our unified weather dataset to reduce data fragmentation and global data divide in Africa.',
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
