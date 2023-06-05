from setuptools import setup, find_packages


VERSION = '0.1.14' 
DESCRIPTION = 'abstracted ML pipelines based on pytorch'
LONG_DESCRIPTION = 'abstracted ML pipelines based on pytorch - mainly for text classification'

# Setting up
setup(
        name="pytorch_nlp_pipeline", 
        version=VERSION,
        author="Lufei Wang",
        author_email="wang.lufei@mayo.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'torch==1.13',
            'transformers>=4.26.0',
            'pandas>=1.5.2',
            'shortuuid',
            'scikit-learn>=1.2.0',
            'contractions'
        ],
        python_requires='>=3.8',
        classifiers= [
           ]
)