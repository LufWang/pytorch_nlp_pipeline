from setuptools import setup, find_packages


VERSION = '0.1.10' 
DESCRIPTION = 'abstracted ML pipelines based on pytorch'
LONG_DESCRIPTION = 'abstracted ML pipelines based on pytorch - mainly for text classification'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pytorch_nlp_pipeline", 
        version=VERSION,
        author="Lufei Wang",
        author_email="wang.lufei@mayo.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'torch==1.12.1',
            'transformers==4.26.0',
            'pandas==1.5.3',
            'shortuuid==1.0.11',
            'scikit-learn==1.2.1',
            'contractions'
        ],
        python_requires='>=3.8',
        classifiers= [
           ]
)