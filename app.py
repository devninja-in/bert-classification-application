import streamlit as st

from config.config import APP_NAME, APP_ICON, APP_LAYOUT
from ui.about_page import render_about_page
from ui.prediction_page import render_prediction_page
from ui.styles import inject_custom_css, main_header, highlight_text
from ui.training_page import render_training_page
from utils.logger import logger


def setup_session_state():
    """Initialize session state variables"""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False


def main():
    """Main application entry point"""
    # Set page configuration
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
    )

    # Inject custom CSS
    inject_custom_css()

    # Initialize session state
    setup_session_state()

    # App title and description
    main_header("BERT Text Classification App")
    highlight_text(
        "This app allows you to train a BERT model for text classification tasks. "
        "You can upload your own labeled data, configure training parameters, and use the trained model to classify new text."
    )

    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Train", "Predict", "About"])

    # Render selected page
    if page == "Upload & Train":
        logger.info("Navigated to Upload & Train page")
        render_training_page()
    elif page == "Predict":
        logger.info("Navigated to Predict page")
        render_prediction_page()
    else:  # About page
        logger.info("Navigated to About page")
        render_about_page()


if __name__ == "__main__":
    try:
        logger.info("Starting BERT Text Classification App")
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
