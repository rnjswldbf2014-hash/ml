import setuptools

setuptools.setup(
    name="rnjswldbf_2014",
    version="0.1.1",
    description="A reinforcement learning library in D",
    author="Jeeyul Kwon",
    license="GPL-2.0",
    packages=["rnjswldbf_2014"],
    package_data={"rnjswldbf_2014": ["*.pyd", "*.dll", "*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
