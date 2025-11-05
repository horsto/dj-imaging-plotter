"""
Datajoint imaging 
Plotter class 
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='djplotter', 
    version='0.1.2',  
    description='Plotter for the datajoint imaging pipeline', 
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/horsto/dj-imaging-plotter',  
    author='Horst Obenhaus',  
    author_email='horst.obenhaus@ntnu.no',  

    keywords='datajoint, imaging-pipeline', 

    packages=find_packages(),  

    python_requires='>=3.6, <4',

    install_requires=[
        'tqdm>=4.55.0',
        'numpy>=1.19.2',
        'matplotlib>=3.3.3',
        'seaborn>=0.11.0',
        'scipy>=1.5.4',
        'scikit-image>=0.17.2',
        'pandas>=1.1.3',
        'cmasher>=1.5.7',
    ], 

    project_urls={  # Optional
        'Source': 'https://github.com/horsto/dj-imaging-plotter',
        'Main':   'https://github.com/kavli-ntnu/dj-moser-imaging',
    },
)
