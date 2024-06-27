from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    desc = f.read()

setup(
    name="fast_gem",
    version="0.0.3",
    description="Efficient and general implementation of Generalized Mean Pooling (GeM)",
    author="Kitsunetic",
    author_email="jh.shim.gg@gmail.com",
    url="https://github.com/Kitsunetic/fast-GeM",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[],
    # entry_points={"console_scripts": ["kitsu=kitsu.main:main"]},
    long_description=desc,
    long_description_content_type="text/markdown",
)
