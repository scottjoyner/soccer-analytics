from setuptools import find_packages, setup

setup(
    name="soccer-analytics",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "rich>=13.7.0",
        "typer>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "soccer-edge=soccer_edge.cli:app",
        ]
    },
)
