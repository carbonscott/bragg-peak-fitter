import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bragg_peak_fitter",
    version="24.02.15",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="A Bragg peak fitter.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/bragg-peak-fitter",
    keywords = ['SFX', 'X-ray', 'Model Fitting', 'LCLS'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
