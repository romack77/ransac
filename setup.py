from setuptools import find_packages
from setuptools import setup

setup(
    name='ransac',
    version='0.0.1',
    description='Robust model fitting for noisy data.',
    license='MIT',
    keywords='ransac',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    install_requires=['numpy', 'scipy'],
    packages=find_packages(exclude=['notebooks']),
    test_suite='ransac',
)
