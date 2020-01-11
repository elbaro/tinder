from setuptools import setup, find_packages


LONG_DESCRIPTION = """
**tinder** provides extra layers and helpers for Pytorch.
"""

setup(
    name="tinder",
    version="0.1.6",
    description="Pytorch helpers and utils",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    url="http://github.com/elbaro/tinder",
    author="elbaro",
    author_email="elbaro@users.noreply.github.com",
    license="MIT",
    packages=find_packages(),
    keywords=["tinder", "pytorch", "torch"],
    zip_safe=False,
    install_requires=[
        "colorama",
        "matplotlib",
        "numpy",
        "pillow-simd",
        "torch",
        "torchvision",
        "tqdm",
    ],
)
