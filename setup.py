from setuptools import setup, find_packages

setup(
    name='AMIVA_F',
    version='0.0.33',
    packages=find_packages(),
    install_requires=[
        "freesasa",
	"biopython",
	"pandas",
	"numpy",
	"scikit-learn",
	"joblib"

    ],
    entry_points={
        'console_scripts': [
            'AMIVA_F = AMIVA_F:main',  # if your package has a command-line script
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
