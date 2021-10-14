from setuptools import setup
with open("ANN_Implementation_DLCVNLP_Demo/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

Project_Name="ANN implementation DLCVNLP demo"
USER_NAME='Vani Bhatia'

setup(
    name="src",
    version="0.0.1",
    author="USER_NAME",
    author_email="vanibhatia13@gmail.com",
    description=" A small package for ANN implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{Project_Name}",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "pandas",
        "seaborn"
    ]
)
