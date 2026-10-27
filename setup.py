from setuptools import setup, find_packages

setup(
    name = 'RelGravPy',
    version = '1.0.0',
    author = 'Zampa Luigi Sante',
    author_email = 'lzampa@ogs.it',
    description = 'A Python package for processing relative gravity data for geophysical applications.',
    packages = find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pyproj',
        'contextily',
    ],
    # Optional fast compilation. Base installs always work without Numba.
    extras_require={'fast': ['numba>=0.59']},
    package_data={'relgravpy': ['data/*.dat', 'data/*.md', 'data/*.txt']},
    include_package_data=True,
    )
