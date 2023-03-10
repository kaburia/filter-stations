from setuptools import setup, find_packages

setup(
    name='filter_stations',
    version='0.1.4',
    packages=find_packages(),
    include_package_data=True,
    description='Making it easier to navigate and clean station data',
    author='Austin Kaburia',
    author_email='kaburiaaustin1@gmail.com',
    url='https://github.com/kaburia/Packaging/tree/main/filter_stations',
    install_requires=[
        'pandas',
        'requests',
        'python-dateutil',
        'argparse',
        'haversine',
        'matplotlib',
        'numpy',
        'IPython',
        'folium'
    ],
    entry_points={
        'console_scripts': [
            'my-script=filter_stations.my_script:main'
        ]
    }
)
