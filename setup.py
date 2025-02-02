from setuptools import setup, find_packages

setup(
    name="distill_cot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "pyyaml>=6.0.1",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
)
