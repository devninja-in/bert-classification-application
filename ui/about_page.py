import streamlit as st

from ui.styles import sub_header


def render_about_page():
    """Render the about page UI"""
    sub_header("About This App")

    st.markdown("""
    ### BERT Text Classification App

    This enterprise-level application allows you to train transformer-based models (DistilBERT/BERT) for text classification tasks and use them to classify new text inputs.

    #### Features:

    - **Upload & Train**: Upload your labeled dataset and train a transformer model with custom parameters
    - **Predict**: Use the trained model to classify new text inputs
    - **Batch Prediction**: Run predictions on multiple texts at once
    - **Save & Load Models**: Save trained models for future use and load them when needed
    - **Label Distribution**: View the distribution of labels in your dataset
    - **Performance Metrics**: View detailed classification reports and confusion matrices

    #### How to Use:

    1. Go to the "Upload & Train" section and upload your CSV file with text and labels
    2. Configure the training parameters and train the model
    3. Once training is complete, go to the "Predict" section to classify new text
    4. Use the Save Model option to store your trained model for future use

    #### About the Model:

    This app uses DistilBERT by default, which is a distilled version of BERT. It's smaller, faster, and often more compatible with various environments while maintaining much of BERT's performance. Like BERT, it's a transformer-based model pre-trained on a large corpus of text and fine-tuned for specific tasks like text classification.

    The full BERT model is also available as an option for training if you need higher accuracy, though it requires more computational resources.

    #### Requirements:

    This app uses:
    - PyTorch
    - Transformers (Hugging Face)
    - Streamlit
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn

    #### Limitations:

    - Training may be slow on CPU-only environments
    - Maximum sequence length is limited to 512 tokens
    - For large datasets, consider training on a more powerful machine outside this app

    #### Project Structure:

    This application is structured as an enterprise-level Python project with:

    - Modular code organization
    - Comprehensive logging
    - Configuration management
    - Separation of concerns (data, models, UI)
    - Test infrastructure
    """)
