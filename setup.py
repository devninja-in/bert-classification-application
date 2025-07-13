from setuptools import setup, find_packages

setup(
    name="bert_text_classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-level BERT text classification application",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bert-text-classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "streamlit>=1.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "python-dotenv>=0.19.0",
    ],
)