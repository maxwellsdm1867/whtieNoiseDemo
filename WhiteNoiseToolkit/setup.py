from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="white-noise-toolkit",
    version="0.1.0",
    author="White Noise Analysis Team",
    author_email="contact@whitenoise.toolkit",
    description="Research-grade Python toolkit for white noise analysis of neuronal responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whitenoise/toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'wnt-test=white_noise_toolkit.examples.installation_test:main',
        ],
    },
    include_package_data=True,
    package_data={
        'white_noise_toolkit': ['config/*.yaml'],
    },
)
