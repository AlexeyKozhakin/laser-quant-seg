from setuptools import setup, find_packages

setup(
    name='laser-segmentation-quantizer',
    version='0.1.0',
    description='Library for accelerating neural networks with quantization for laser beam segmentation on conveyor belts.',
    author='Alexey Kozhakin',
    author_email='alexeykozhakin@gmail.com',
    url='https://github.com/AlexeyKozhakin/laser-quant-seg',
    packages=find_packages(),
    install_requires=[
        'tensorflow',  # Добавьте необходимые зависимости
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
