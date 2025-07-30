# BERT Text Classification Application

An enterprise-level text classification application using BERT transformers with a modern Streamlit web interface.

## Overview

This application provides a comprehensive solution for training and deploying BERT models for text classification tasks. Built with Streamlit and leveraging Hugging Face's transformers library, it offers an intuitive web interface for both technical and non-technical users.

## Features

- **Interactive Training**: Upload CSV datasets and train customized BERT models through a user-friendly web interface
- **Real-time Prediction**: Classify new text using trained models with instant results
- **Batch Processing**: Run predictions on multiple texts simultaneously
- **Model Management**: Save, load, and manage trained models with automatic persistence
- **Data Visualization**: View data distributions, confusion matrices, and model performance metrics
- **Custom Styling**: Modern UI with custom CSS styling and responsive design
- **Comprehensive Logging**: Detailed application logging for debugging and monitoring
- **Configurable Parameters**: Adjustable training hyperparameters and model settings

## Project Structure

```
bert-classification-application/
├── app.py                    # Main Streamlit application entry point
├── config/                   # Configuration settings
│   ├── __init__.py
│   └── config.py            # App configuration and constants
├── data/                    # Data handling utilities
│   ├── __init__.py
│   └── data_loader.py       # CSV loading and preprocessing
├── models/                  # Model definition and training
│   ├── __init__.py
│   ├── model.py            # BERT classifier implementation
│   └── training.py         # Training logic and utilities
├── ui/                     # User interface components
│   ├── __init__.py
│   ├── about_page.py       # About page UI
│   ├── prediction_page.py  # Prediction interface
│   ├── styles.py           # Custom CSS and styling
│   └── training_page.py    # Training interface
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── visualization.py    # Charts and plots
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_model.py       # Model tests
├── trained_models/         # Saved model files (auto-created)
├── logs/                   # Application logs (auto-created)
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation
├── Containerfile          # Container deployment
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd bert-classification-application
```

2. **Create and activate a virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:8501
```

### Training a Model

1. **Navigate to "Upload & Train" page**
2. **Upload your dataset** (CSV format with text and label columns)
3. **Select your text and label columns** from the dropdown menus
4. **Configure training parameters:**
   - Model name (default: distilbert-base-uncased)
   - Maximum sequence length
   - Batch size
   - Number of epochs
   - Learning rate
5. **Click "Train Model"** and monitor the progress
6. **View training results** including confusion matrix and classification report

### Making Predictions

1. **Navigate to "Predict" page**
2. **Load a previously trained model** from the dropdown
3. **Enter text** for classification
4. **View predictions** with confidence scores
5. **Use batch prediction** for multiple texts at once

## Data Format

Your CSV file should contain at least two columns:
- **Text column**: Contains the text data to be classified
- **Label column**: Contains the target labels/categories

Example:
```csv
text,label
"This movie is amazing!",positive
"I didn't like this product",negative
"Great customer service",positive
```

## Model Support

The application supports any BERT-based model from Hugging Face Hub:
- `distilbert-base-uncased` (default, faster training)
- `bert-base-uncased`
- `roberta-base`
- `albert-base-v2`
- And many more...

## Configuration

Key settings can be modified in `config/config.py`:

- **Model settings**: Default model, max length, batch size
- **Training parameters**: Epochs, learning rate, test split ratio
- **UI settings**: Colors, layout, styling
- **Logging**: Log level and format

## Dependencies

- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities
- **matplotlib/seaborn**: Data visualization
- **python-dotenv**: Environment variable management

See `requirements.txt` for specific versions.

## Container Deployment

Build and run using the provided Containerfile:

```bash
# Build container
podman build -t bert-classifier .

# Run container
podman run -p 8501:8501 bert-classifier
```

## Logging

Application logs are automatically saved to `logs/app.log`. Log level can be configured via the `LOG_LEVEL` environment variable or in `config/config.py`.

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open an issue in the GitHub repository.
