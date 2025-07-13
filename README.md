# BERT Text Classification Project

An enterprise-level text classification application using BERT transformers.

## Overview

This application provides a user-friendly interface for training and using BERT models to classify text. It's built with Streamlit and leverages Hugging Face's transformers library to provide powerful text classification capabilities.

## Features

- **Training**: Upload labeled datasets and train customized BERT models
- **Prediction**: Classify new text using trained models
- **Batch Processing**: Run predictions on multiple texts simultaneously
- **Model Management**: Save, load, and manage trained models
- **Visualization**: View data distributions and model performance metrics

## Project Structure

```
bert_text_classifier/
├── config/        # Configuration settings
├── data/          # Sample and user data
├── logs/          # Application logs
├── models/        # Saved model files
├── src/           # Source code
│   ├── data/      # Data handling utilities
│   ├── models/    # Model definition and training
│   ├── utils/     # Helper utilities
│   └── ui/        # User interface components
└── tests/         # Test suite
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bert-text-classifier.git
cd bert-text-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python -m src.app
```

2. Open your browser and go to http://localhost:8501

3. Upload your dataset (CSV format with text and label columns)

4. Configure training parameters and train your model

5. Use the trained model to make predictions

## Sample Data

A sample dataset is provided in the `data/` directory for demonstration purposes.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.10+
- Streamlit 1.0+
- See requirements.txt for full dependencies

## Logging

Logs are stored in the `logs/` directory. The log level can be configured in `config/config.py`.

## License

MIT License