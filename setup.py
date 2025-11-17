from setuptools import setup, find_packages

setup(
    name="swiss-trading-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.32",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    author="David Jaggi",
    description="A multi-agent system for trading Swiss Market Index stocks",
)
