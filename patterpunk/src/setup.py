from setuptools import find_packages, setup
from pathlib import Path


VERSION = "0.0.19"

current_dir = Path(__file__).parent
requirements_path = current_dir / "requirements.txt"


setup(
    name="patterpunk",
    description="A simple library to interact with various LLM providers",
    long_description=open(current_dir / "patterpunk" / "README.md").read(),
    long_description_content_type="text/markdown",
    version=VERSION,
    packages=find_packages(exclude=["tests", "bin", "dist"]),
    url="https://github.com/trb/patterpunk",
    author="Thomas Rubbert",
    author_email="thomas.rubbert@yahoo.de",
    license="MPL-2.0",
    include_package_data=True,
    install_requires=[
        # This reads the requirements from the requirements.txt file
        line.strip()
        for line in open(requirements_path, "r")
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
