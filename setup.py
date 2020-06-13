import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='scGeneFit',  
     version='1.0.0',
     author="Soledad Villar",
     author_email="soledad.villar@nyu.edu",
     description="Genetic marker selection with linear programming",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/solevillar/scGeneFit-python",
     packages=setuptools.find_packages(),
     include_package_data=True,
     install_requires=['numpy', 'matplotlib', 'scipy', 'sklearn'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
 )
