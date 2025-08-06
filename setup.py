from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="cif-pipeline",
    version="0.1.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="A sandbox for instruction fine-tuning with quality control and enhancement features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/cif-pipeline",
    packages=find_packages(include=["modules*", "infer*", "recipe*", "tripwires*"]),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="instruction-tuning, language-models, fine-tuning, nlp, reinforcement-learning, data-enhancement",
    entry_points={
        "console_scripts": [
            "cif-pipeline=run_pipeline:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "full": [
            "accelerate>=0.20.0",
            "wandb>=0.13.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
)
