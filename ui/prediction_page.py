import os
import streamlit as st
import pandas as pd

from config.config import MODELS_DIR
from data.data_loader import DataLoader
from models.model import BERTClassifier
from utils.logger import logger
from ui.styles import sub_header, info_text, success_text, warning_text


def render_prediction_page():
    """Render the prediction page UI"""
    sub_header("Make Predictions")

    # Check if a model is loaded
    if not st.session_state.get('trained', False):
        sub_header("Load a Saved Model")

        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        saved_model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pth')]
        saved_models = [f.split('_model.pth')[0] for f in saved_model_files]

        if saved_models:
            selected_model = st.selectbox("Select a saved model for prediction", saved_models)
            if st.button("Load Model"):
                try:
                    model_path = os.path.join(MODELS_DIR, f"{selected_model}_model.pth")
                    components_path = os.path.join(MODELS_DIR, f"{selected_model}_components.pkl")

                    classifier = BERTClassifier.load(model_path, components_path)

                    # Store in session state
                    st.session_state.classifier = classifier
                    st.session_state.trained = True

                    success_text(f"Model '{selected_model}' loaded successfully!")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    st.error(f"Error loading model: {str(e)}")
        else:
            info_text("No saved models found. Please train a model first.")

    if st.session_state.get('trained', False) and hasattr(st.session_state, 'classifier'):
        classifier = st.session_state.classifier

        # Text input for prediction
        text_input = st.text_area("Enter text to classify:", height=150)

        if st.button("Classify"):
            if text_input:
                with st.spinner("Classifying..."):
                    try:
                        # Make prediction
                        predicted_labels, probabilities = classifier.predict([text_input])
                        predicted_label = predicted_labels[0]
                        probs = probabilities[0]

                        # Get class names
                        classes = classifier.label_encoder.classes_

                        # Display result
                        success_text(f"Predicted class: **{predicted_label}**")

                        # Display probabilities
                        sub_header("Class Probabilities")

                        # Create DataFrame for display
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probs
                        })

                        # Sort by probability (descending)
                        prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)

                        # Display as bar chart
                        st.bar_chart(prob_df.set_index('Class'))
                        st.write(prob_df)

                    except Exception as e:
                        logger.error(f"Error during prediction: {str(e)}")
                        st.error(f"An error occurred during prediction: {str(e)}")
            else:
                warning_text("Please enter some text to classify.")

        # Batch prediction
        sub_header("Batch Prediction")
        uploaded_predict_file = st.file_uploader(
            "Upload file for batch prediction (CSV with a text column)",
            type=["csv"]
        )

        if uploaded_predict_file is not None:
            # Load prediction data
            data_loader = DataLoader()
            predict_df = data_loader.load_csv_from_upload(uploaded_predict_file)

            if predict_df is not None:
                st.write("Preview:")
                st.write(predict_df.head())

                # Select text column
                text_col = st.selectbox("Select text column for prediction", predict_df.columns)

                if st.button("Run Batch Prediction"):
                    with st.spinner("Running batch prediction..."):
                        try:
                            # Get texts for prediction
                            texts = predict_df[text_col].tolist()

                            # Make predictions
                            predicted_labels, _ = classifier.predict(texts)

                            # Add predictions column
                            predict_df['predicted_class'] = predicted_labels

                            # Display results
                            success_text("Batch prediction completed!")
                            st.write(predict_df)

                            # Download link for results
                            st.markdown(
                                data_loader.create_download_link(
                                    predict_df,
                                    'prediction_results.csv',
                                    'Download prediction results'
                                ),
                                unsafe_allow_html=True
                            )

                        except Exception as e:
                            logger.error(f"Error during batch prediction: {str(e)}")
                            st.error(f"An error occurred during batch prediction: {str(e)}")
            else:
                st.error("Failed to load the uploaded file. Please check if it's a valid CSV.")
    else:
        warning_text("Please train a model first in the 'Upload & Train' section or load a saved model.")