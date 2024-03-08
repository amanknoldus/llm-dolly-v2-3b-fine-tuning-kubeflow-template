from typing import List

from setuptools import setup


def read_requirements() -> List[str]:
    fpath = 'requirements.txt'
    with open(fpath, 'r') as fd:
        candidate_reqs = (line.split('#')[0].strip() for line in fd)
        return [req for req in candidate_reqs if req]


if __name__ == "__main__":
    with open('./README.md') as f:
        readme = f.read()

    requirements = read_requirements()

    with open("version.txt", "r") as f:
        version = f.read().strip()

    setup(
        name="pipeline_name",
        version=version,
        description="your_pipeline_description",
        long_description=readme,
        author="author_name",
        author_email="author_email",
        packages=['utils'],
        python_requires=">=3.8",
        install_requires=requirements,
    )
