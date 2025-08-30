import os
import streamlit as st
import pandas as pd

from config.config import MODELS_DIR
from data.data_loader import DataLoader
from models.model import BERTClassifier, BERTMultiLabelClassifier
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

        # Get both multi-class and multi-label models
        multiclass_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pth')]
        multilabel_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_multilabel_model.pth')]
        
        saved_models = []
        for f in multiclass_files:
            model_name = f.split('_model.pth')[0]
            saved_models.append((model_name, 'Multi-Class'))
        
        for f in multilabel_files:
            model_name = f.split('_multilabel_model.pth')[0]
            saved_models.append((model_name, 'Multi-Label'))

        if saved_models:
            # Create display options with model type
            model_options = [f"{name} ({model_type})" for name, model_type in saved_models]
            selected_option = st.selectbox("Select a saved model for prediction", model_options)
            
            if st.button("Load Model"):
                try:
                    # Extract model name and type
                    selected_index = model_options.index(selected_option)
                    selected_model, model_type = saved_models[selected_index]
                    
                    if model_type == 'Multi-Label':
                        model_path = os.path.join(MODELS_DIR, f"{selected_model}_multilabel_model.pth")
                        components_path = os.path.join(MODELS_DIR, f"{selected_model}_multilabel_components.pkl")
                        classifier = BERTMultiLabelClassifier.load(model_path, components_path)
                    else:
                        model_path = os.path.join(MODELS_DIR, f"{selected_model}_model.pth")
                        components_path = os.path.join(MODELS_DIR, f"{selected_model}_components.pkl")
                        classifier = BERTClassifier.load(model_path, components_path)

                    # Store in session state
                    st.session_state.classifier = classifier
                    st.session_state.trained = True
                    st.session_state.model_type = model_type

                    success_text(f"{model_type} model '{selected_model}' loaded successfully!")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    st.error(f"Error loading model: {str(e)}")
        else:
            info_text("No saved models found. Please train a model first.")

    if st.session_state.get('trained', False) and hasattr(st.session_state, 'classifier'):
        classifier = st.session_state.classifier
        
        # Determine model type - first from session state, then from classifier type
        model_type = st.session_state.get('model_type', 'Multi-Class')
        
        # Fallback: detect from classifier type if session state is wrong
        if hasattr(classifier, 'label_binarizer'):
            model_type = 'Multi-Label'
        elif hasattr(classifier, 'label_encoder'):
            model_type = 'Multi-Class'
        
        # Update session state if it was wrong
        st.session_state.model_type = model_type

        # Show model type info with classifier details
        classifier_type = type(classifier).__name__
        if model_type == 'Multi-Label':
            st.info(f"🏷️ **Active Model**: {model_type} - Can predict multiple labels per text")
            st.caption(f"Classifier type: {classifier_type} | Has label_binarizer: {hasattr(classifier, 'label_binarizer')}")
        else:
            st.info(f"🏷️ **Active Model**: {model_type} - Predicts single label per text")
            st.caption(f"Classifier type: {classifier_type} | Has label_encoder: {hasattr(classifier, 'label_encoder')}")

        # Text input for prediction
        text_input = st.text_area("Enter text to classify:", height=150)

        if st.button("Classify"):
            if text_input:
                with st.spinner("Classifying..."):
                    try:
                        # Make prediction
                        predicted_labels, probabilities = classifier.predict([text_input])
                        
                        if model_type == 'Multi-Label':
                            # Multi-label prediction handling
                            predicted_label_list = predicted_labels[0]
                            probs = probabilities[0]

                            # Get class names
                            if hasattr(classifier, 'label_binarizer'):
                                classes = classifier.label_binarizer.classes_
                            else:
                                classes = list(range(len(probs)))

                            # Display results
                            if predicted_label_list:
                                labels_str = ", ".join(predicted_label_list)
                                success_text(f"Predicted labels: **{labels_str}**")
                            else:
                                warning_text("No labels exceeded the prediction threshold")

                            # Display probabilities
                            sub_header("Label Probabilities")

                            # Create DataFrame for display
                            prob_df = pd.DataFrame({
                                'Label': classes,
                                'Probability': probs
                            })

                            # Sort by probability (descending)
                            prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                            
                            # Highlight predicted labels
                            prob_df['Predicted'] = prob_df['Label'].isin(predicted_label_list)

                            # Display as bar chart
                            st.bar_chart(prob_df.set_index('Label')['Probability'])
                            st.write(prob_df)

                            # Show threshold info
                            st.caption(f"Threshold: {classifier.threshold:.2f} - Labels above this threshold are considered positive predictions")

                        else:
                            # Multi-class prediction handling (original logic)
                            predicted_label = predicted_labels[0]
                            probs = probabilities[0]

                            # Get class names
                            if hasattr(classifier, 'label_encoder'):
                                classes = classifier.label_encoder.classes_
                            else:
                                classes = list(range(len(probs)))

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
                            predicted_labels, probabilities = classifier.predict(texts)

                            if model_type == 'Multi-Label':
                                # Multi-label batch prediction
                                # Convert list of lists to strings for CSV export
                                predict_df['predicted_labels'] = [", ".join(labels) if labels else "No labels" for labels in predicted_labels]
                                predict_df['probabilities'] = probabilities
                                
                                # Show label statistics
                                sub_header("Prediction Statistics")
                                all_predicted_labels = [label for labels in predicted_labels for label in labels]
                                if all_predicted_labels:
                                    label_counts = pd.Series(all_predicted_labels).value_counts()
                                    st.bar_chart(label_counts)
                                else:
                                    st.warning("No labels were predicted above the threshold")
                            else:
                                # Multi-class batch prediction (original)
                                predict_df['predicted_class'] = predicted_labels
                                predict_df['probabilities'] = probabilities

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