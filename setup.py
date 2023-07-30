import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="franklabnwb",
    version="0.0.1",
    author="Frank Lab members",
    author_email="loren@phy.ucsf.edu",
    description="NWB helper code for Loren Frank's lab at UCSF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/franklabnwb",
    packages=setuptools.find_packages(),
    package_data={'franklabnwb':["*.yaml"]},
    install_requires=[
        'pynwb',
        'hdmf',
        'pandas',
        'networkx',
        'python-intervals',
        'matplotlib',
        'numpy',
        'scipy',
        'python-dateutil'
    ],
    entry_points='''
        [console_scripts]
        create_franklab_spec=franklabnwb.create_franklab_spec:main
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
