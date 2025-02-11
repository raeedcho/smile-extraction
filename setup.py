from setuptools import find_packages, setup

setup(
    name="smile_extract",
    packages=find_packages(),
    version="0.1.0",
    description="A package to extract and process neural and behavioral data collected by monkey host in the Batista lab",
    author="Raeed Chowdhury",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "mat73",
    ],
)
