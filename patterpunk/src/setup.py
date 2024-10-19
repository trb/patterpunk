from setuptools import find_packages, setup
from pathlib import Path


VERSION = "0.0.6"

current_dir = Path(__file__).parent
requirements_path = current_dir / "requirements.txt"


setup(
    name="patterpunk",
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
        "License :: OSI Approved :: Mozilla Public License 2.0",
        "Intended Audience :: Developers",
    ],
)
