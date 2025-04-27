from setuptools import setup, find_packages

setup(
    name='news-ai-endpoint',
    version='0.1.0',
    description='A machine learning API service for deploying models.',
    author='agharsallah',
    author_email='abderrahmen.gharsallah@yahoo.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'fastapi',
        'uvicorn',
        'sentence-transformers',
        'pydantic',
        'pydantic-settings',
        'python-dotenv',
        'torch',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'mypy',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Framework :: FastAPI',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12.3',
)