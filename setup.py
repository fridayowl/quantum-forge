from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-forge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Essential tools for quantum computing: circuit optimization, error mitigation, and quantum algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-forge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qiskit>=0.39.0",
        "qiskit-aer>=0.11.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "chemistry": [
            "pyscf>=2.0.0",
            "openfermion>=1.3.0",
        ],
        "ml": [
            "torch>=1.12.0",
            "pennylane>=0.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qforge=quantum_forge.cli:main",
        ],
    },
)
