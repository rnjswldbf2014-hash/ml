import pathlib
import setuptools

this_dir = pathlib.Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="rnjswldbf_2014",
    version="0.1.2",
    description="A reinforcement learning / supervised learning library implemented in D, exposed as a Python extension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jeeyul Kwon",
    url="https://github.com/rnjswldbf2014-hash/ml",
    license="GPL-2.0",
    packages=["rnjswldbf_2014"],
    package_data={"rnjswldbf_2014": ["*.pyd", "*.dll", "*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
