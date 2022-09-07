""" setuptools-based setup module. """

from setuptools import setup, find_packages

setup(
    name="entrainment",
    description="Deep learning model for vocal entrainment",
    url="https://github.com/meghavarshini/unsupervised-learning-entrainment",
    packages = find_packages(),

    keywords="vocalic feature modelling",
    # zip_safe=False,
    install_requires=[
        "wheel",
        "torch==1.9.0+cu111",
        "torchvision==0.10.0+cu111",
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "tqdm",
        "webvtt-py",
        "transformers",
        "h5py",
        "kaldi-io==0.9.4",
        "scipy"
    ],

    python_requires=">=3.8",
)
