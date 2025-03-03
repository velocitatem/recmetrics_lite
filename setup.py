from setuptools import setup, find_packages

setup(
    name="recmetrics-lite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
    ],
    extras_require={
        "plots": ["plotly"],
    },
    description="A streamlined library for recommender system evaluation metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Daniel Rosel",
    author_email="daniel@alves.world",
    url="https://github.com/velocitatem/recmetrics_lite",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)
