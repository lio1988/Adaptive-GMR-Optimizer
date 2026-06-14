from setuptools import setup, find_packages

setup(
    name="adaptive-gmr-optimizer",
    version="0.1.0",
    description="A robust, self-healing optimizer for LLMs using Geman-McClure estimation.",
    author="Lio1988",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
