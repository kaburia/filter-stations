from setuptools import setup

setup(
    name='filter_stations',
    version='0.1',
    description='Making it easier to navigate and clean station data',
    author='Austin Kaburia',
    author_email='kaburiaaustin1@gmail.com',
    url='https://github.com/yourusername/my-package',
    packages=['filter_stations'],
    install_requires=[
        'pandas',
        'requests',
        'python-dateutil',
        'argparse',
    ],
    entry_points={
        'console_scripts': [
            'my-script=my_package.my_script:main'
        ]
    }
)
