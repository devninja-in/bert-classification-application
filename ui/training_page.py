import os
import streamlit as st

from config.config import MODELS_DIR, CLASSIFICATION_TYPES, DEFAULT_CLASSIFICATION_TYPE
from data.data_loader import DataLoader
from models.model import BERTClassifier, BERTMultiLabelClassifier
from models.training import train_model
from utils.visualization import plot_confusion_matrix, format_classification_report
from utils.logger import logger
from ui.styles import sub_header, info_text, success_text, warning_text


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

            # Classification Type Selection (before preprocessing)
            sub_header("Classification Type")
            classification_type = st.radio(
                "Select classification type:",
                CLASSIFICATION_TYPES,
                index=CLASSIFICATION_TYPES.index(DEFAULT_CLASSIFICATION_TYPE),
                help="Multi-Class: Single label prediction (e.g., sentiment). Multi-Label: Multiple labels per text (e.g., topics)"
            )
            
            # Show multi-label format info
            if classification_type == "Multi-Label":
                st.info("📝 **Multi-Label Format**: Your label column should contain multiple labels separated by commas or in list format. Examples: 'Sports,Football' or \"['Sports','Football']\"")
                
                # Show sample multi-label data format
                with st.expander("View Multi-Label Data Format Examples"):
                    st.markdown("""
                    **Supported formats:**
                    - Comma-separated: `Sports, Football, NFL`
                    - List format: `['Sports', 'Football', 'NFL']`
                    - Single label: `Sports`
                    
                    **Sample Data:**
                    ```
                    text,labels
                    "Great football game!","Sports,Football"
                    "Movie was amazing","Entertainment,Movies,Drama"
                    "Stock market news","Finance"
                    ```
                    """)
                    
                # Threshold setting for multi-label
                threshold = st.slider(
                    "Multi-label prediction threshold", 
                    0.1, 0.9, 0.5, 0.05,
                    help="Probability threshold for assigning labels in multi-label classification"
                )
            else:
                st.info("📝 **Multi-Class Format**: Your label column should contain a single label per row. Example: 'positive', 'negative', 'neutral'")

            # Preprocess data (after classification type is selected)
            df = data_loader.preprocess_data(df, label_column, classification_type=classification_type) 

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
                selected_option = st.selectbox("Select saved model", model_options)
                
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
                info_text("No saved models found.")

            # Train button
            if st.button("Train Model"):
                # Clear any previously loaded model from session state to avoid confusion
                if 'classifier' in st.session_state:
                    del st.session_state.classifier
                if 'trained' in st.session_state:
                    st.session_state.trained = False
                if 'model_type' in st.session_state:
                    del st.session_state.model_type
                    
                with st.spinner(f"Training {classification_type} BERT model... This may take several minutes."):
                    try:
                        # Set up progress display
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Train model with classification type
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
                            classification_type=classification_type,
                            progress_bar=progress_bar.progress,
                            status_text=status_text.text
                        )

                        # Store in session state
                        st.session_state.classifier = classifier
                        st.session_state.trained = True
                        st.session_state.model_type = classification_type  # Set model_type based on classification_type
                        # Store training results in session state for persistence
                        st.session_state.training_results = {
                            'report': report,
                            'cm': cm,
                            'class_names': class_names,
                            'label_column': label_column,
                            'df': df,
                            'classification_type': classification_type
                        }

                        # Display results
                        success_text("Training completed!")

                    except Exception as e:
                        logger.error(f"Error during training: {str(e)}")
                        st.error(f"An error occurred during training: {str(e)}")

            # Display training results if they exist in session state
            if st.session_state.get('training_results'):
                results = st.session_state.training_results
                classification_result_type = results.get('classification_type', 'Multi-Class')
                
                if classification_result_type == 'Multi-Label':
                    # Multi-label specific metrics
                    sub_header("Multi-Label Classification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hamming Loss", f"{results['report'].get('hamming_loss', 0):.4f}")
                    with col2:
                        st.metric("Jaccard Score", f"{results['report'].get('jaccard_score', 0):.4f}")
                    with col3:
                        st.metric("Accuracy", f"{results['report'].get('accuracy', 0):.4f}")
                    
                    # Show available labels
                    sub_header("Available Labels")
                    st.write(results['class_names'])
                    
                else:
                    # Multi-class results (original)
                    # Show classification report
                    sub_header("Classification Report")
                    report_df = format_classification_report(results['report'])
                    st.write(report_df)

                    # Plot confusion matrix
                    sub_header("Confusion Matrix")
                    cm_image = plot_confusion_matrix(results['cm'], results['class_names'])
                    st.markdown(f'<img src="data:image/png;base64,{cm_image}" width="600">', unsafe_allow_html=True)

                # Class distribution (for both types)
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
        
        # Sample data type selection
        sample_type = st.radio(
            "Sample data type:",
            ["Multi-Class", "Multi-Label"],
            key="sample_type"
        )
        
        sample_data = DataLoader.get_sample_data(sample_type)
        st.write(sample_data)

        # Create download link for sample data
        filename = f'sample_{sample_type.lower().replace("-", "_")}_data.csv'
        st.markdown(
            DataLoader.create_download_link(sample_data, filename, f'Download sample {sample_type} data CSV'),
            unsafe_allow_html=True
        )