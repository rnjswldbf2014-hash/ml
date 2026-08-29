import setuptools

setuptools.setup(
    name="rnjswldbf_2014",
    version="0.1.0",
    description="A reinforcement learning library in D",
    author="沅뚯???,
    license="GPL-2.0",
    packages=setuptools.find_packages(),
    # Since ml.pyd is a precompiled extension, we can include it as package data
    package_data={"": ["*.pyd", "*.dll", "*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
