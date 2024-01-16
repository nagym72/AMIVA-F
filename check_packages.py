#!/usr/bin/env python3

import pkg_resources

packages = ['freesasa', 'biopython', 'pandas', 'numpy', 'scikit-learn', 'joblib']

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: not installed")
