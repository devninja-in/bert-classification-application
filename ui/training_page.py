import os
import streamlit as st

from config.config import MODELS_DIR
from data.data_loader import DataLoader
from models.model import BERTClassifier
from models.training import train_model
from utils.visualization import plot_confusion_matrix, format_classification_report
from utils.logger import logger
from ui.styles import sub_header, info_text, success_text


def render_training_page():
    """Render the training page UI"""
    sub_header("Upload Data & Train Model")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Load data
        data_loader = DataLoader()
        df = data_loader.load_csv_from_upload(uploaded_file)

        if df is not None:
             # Default label column
            st.write("Dataset Preview:")
            st.write(df.head())

            # Get column names
            column_names = df.columns.tolist()

            # Select columns for text and label
            col1, col2 = st.columns(2)
            with col1:
                label_column = st.selectbox("Select label column", column_names)
            with col2:
                text_column = st.selectbox("Select text column", column_names)
                

            # Ensure columns are selected (they should be by default)
            text_column = text_column or column_names[0]
            label_column = label_column or column_names[0]

            # Preprocess data
            df = data_loader.preprocess_data(df, label_column) 

            # Show count by label
            col_table, col_chart = st.columns(2)
            with col_table:
                sub_header("Label Count Distribution")
                label_counts = data_loader.get_label_distribution(df, label_column)
                st.write(label_counts)

            with col_chart:
                sub_header("Label Distribution Chart")
                st.bar_chart(df[label_column].value_counts())

            # Training parameters
            sub_header("Model Training Parameters")

            # First row of parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
            with col2:
                epochs = st.slider("Number of epochs", 1, 10, 3)

            # Second row of parameters
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
            with col2:
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.00001, 0.0001, 0.001, 0.01],
                    value=0.0001,
                    format_func=lambda x: f"{x:.5f}"
                )

            # Third row of parameters
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.select_slider("Max sequence length", options=[64, 128, 256, 512], value=128)
            with col2:
                model_name = st.selectbox(
                    "Pretrained model",
                    ["distilbert-base-uncased", "bert-base-uncased"],
                    index=0
                )

            # Model Loading Section
            sub_header("Load Existing Model")

            # Create models directory if it doesn't exist
            os.makedirs(MODELS_DIR, exist_ok=True)

            saved_model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pth')]
            saved_models = [f.split('_model.pth')[0] for f in saved_model_files]

            if saved_models:
                selected_model = st.selectbox("Select saved model", saved_models)
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
                info_text("No saved models found.")

            # Train button
            if st.button("Train Model"):
                with st.spinner("Training BERT model... This may take several minutes."):
                    try:
                        # Set up progress display
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Train model
                        classifier, report, cm, class_names = train_model(
                            df=df,
                            text_column=text_column,
                            label_column=label_column,
                            test_size=test_size,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            max_length=max_length,
                            model_name=model_name,
                            progress_bar=progress_bar.progress,
                            status_text=status_text.text
                        )

                        # Store in session state
                        st.session_state.classifier = classifier
                        st.session_state.trained = True
                        # Store training results in session state for persistence
                        st.session_state.training_results = {
                            'report': report,
                            'cm': cm,
                            'class_names': class_names,
                            'label_column': label_column,
                            'df': df
                        }

                        # Display results
                        success_text("Training completed!")

                    except Exception as e:
                        logger.error(f"Error during training: {str(e)}")
                        st.error(f"An error occurred during training: {str(e)}")

            # Display training results if they exist in session state
            if st.session_state.get('training_results'):
                results = st.session_state.training_results
                
                # Show classification report
                sub_header("Classification Report")
                report_df = format_classification_report(results['report'])
                st.write(report_df)

                # Plot confusion matrix
                sub_header("Confusion Matrix")
                cm_image = plot_confusion_matrix(results['cm'], results['class_names'])
                st.markdown(f'<img src="data:image/png;base64,{cm_image}" width="600">', unsafe_allow_html=True)

                # Class distribution
                sub_header("Class Distribution")
                class_counts = results['df'][results['label_column']].value_counts()
                st.bar_chart(class_counts)

            # Save model section - moved outside training block for persistence
            if st.session_state.get('trained', False) and st.session_state.get('classifier'):
                sub_header("Save Trained Model")
                model_name_input = st.text_input("Enter a name for your model:", "bert_classifier")
                if st.button("Save Model"):
                    try:
                        classifier = st.session_state.classifier
                        model_path, components_path = classifier.save(model_name_input)
                        success_text(f"Model saved successfully as '{model_name_input}'!")
                        logger.info(f"Model saved successfully: {model_path}, {components_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {str(e)}")
                        st.error(f"Error saving model: {str(e)}")

        else:
            st.error("Failed to load the uploaded file. Please check if it's a valid CSV.")
    else:
        info_text("Please upload a CSV file to begin.")

        # Sample data example
        sub_header("Sample Data Format")
        sample_data = DataLoader.get_sample_data()
        st.write(sample_data)

        # Create download link for sample data
        st.markdown(
            DataLoader.create_download_link(sample_data, 'sample_data.csv', 'Download sample data CSV'),
            unsafe_allow_html=True
        )