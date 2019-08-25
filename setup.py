from setuptools import setup

setup(
    name='jerry_says',
    version='0.1',
    maintainer='Arbin Timilsina',
    maintainer_email='arbin.timilsina@gmail.com',
    platforms=['any'],
    description='Predict how Jerry Seinfeld would finish an incomplete sentence.',
    packages=['jerry_says'],
    include_package_data=True,
    install_requires=[
        'nltk', 'spacy', 'pytest', 'tqdm', 'torch', 'torchtext', 'flask'
    ],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        'console_scripts': [
            'train-jerry = jerry_says.cli_train_jerry:main',
            'serve-jerry = jerry_says.cli_server:main',
        ]
    },
)
