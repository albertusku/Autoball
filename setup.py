from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="autoball",
    version="0.1.0",
    description="Seguimiento de balón de baloncesto con visión artificial",
    author="Alberto Ruiz",
    packages=find_packages(),  
    install_requires=requirements,
    python_requires=">=3.8",
)
