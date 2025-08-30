import pandas as pd
from typing import Optional, List, Union
import base64
import ast
import numpy as np

from utils.logger import logger


class DataLoader:
    """Data loading and preprocessing utilities"""

    @staticmethod
    def load_csv(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame or None if error
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None

    @staticmethod
    def load_csv_from_upload(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded CSV file

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            DataFrame or None if error
        """
        try:
            df = pd.read_csv(uploaded_file)
            logger.info(f"Successfully loaded uploaded data: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading uploaded data: {str(e)}")
            return None

    @staticmethod
    def preprocess_data(df: pd.DataFrame, label_column: str, min_samples: int = 5, classification_type: str = "Multi-Class") -> pd.DataFrame:
        """
        Preprocess data for training

        Args:
            df: Input DataFrame
            label_column: Name of the label column
            min_samples: Minimum number of samples per class
            classification_type: Type of classification ('Multi-Class' or 'Multi-Label')

        Returns:
            Preprocessed DataFrame
        """
        if classification_type == "Multi-Label":
            # Multi-label preprocessing
            return DataLoader._preprocess_multilabel_data(df, label_column, min_samples)
        else:
            # Multi-class preprocessing (original)
            filtered_df = df[df.groupby(label_column)[label_column].transform('count') >= min_samples]
            logger.info(f"Preprocessing data: {len(df)} -> {len(filtered_df)} rows after filtering")
            return filtered_df

    @staticmethod
    def _preprocess_multilabel_data(df: pd.DataFrame, label_column: str, min_samples: int = 5) -> pd.DataFrame:
        """
        Preprocess multi-label data for training

        Args:
            df: Input DataFrame
            label_column: Name of the label column
            min_samples: Minimum number of samples per label

        Returns:
            Preprocessed DataFrame
        """
        # Parse multi-label strings and count labels
        all_labels = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            labels = DataLoader.parse_multilabel_string(row[label_column])
            if labels:  # Only keep rows with at least one label
                all_labels.extend(labels)
                valid_rows.append(idx)
        
        # Filter DataFrame to only valid rows
        filtered_df = df.loc[valid_rows].copy()
        
        # Count label occurrences
        label_counts = pd.Series(all_labels).value_counts()
        
        # Find labels with sufficient samples
        valid_labels = set(label_counts[label_counts >= min_samples].index)
        
        if not valid_labels:
            logger.warning("No labels have sufficient samples after filtering")
            return filtered_df
        
        # Filter out rows that don't contain any valid labels
        def has_valid_labels(label_str):
            labels = DataLoader.parse_multilabel_string(label_str)
            return any(label in valid_labels for label in labels)
        
        final_df = filtered_df[filtered_df[label_column].apply(has_valid_labels)]
        
        logger.info(f"Multi-label preprocessing: {len(df)} -> {len(final_df)} rows after filtering")
        logger.info(f"Valid labels ({len(valid_labels)}): {sorted(valid_labels)}")
        
        return final_df

    @staticmethod
    def parse_multilabel_string(label_str: Union[str, float]) -> List[str]:
        """
        Parse multi-label string into list of labels
        Supports formats: "label1,label2", "['label1','label2']", etc.
        """
        if pd.isna(label_str) or label_str == "":
            return []
        
        label_str = str(label_str)
        
        # If it's already a list-like string, try to evaluate it
        if label_str.startswith('[') and label_str.endswith(']'):
            try:
                return ast.literal_eval(label_str)
            except:
                # Fallback to simple parsing
                return [l.strip().strip("'\"") for l in label_str[1:-1].split(',') if l.strip()]
        
        # Simple comma-separated format
        if ',' in label_str:
            return [l.strip() for l in label_str.split(',') if l.strip()]
        
        # Single label
        return [label_str.strip()]

    @staticmethod
    def validate_multilabel_data(df: pd.DataFrame, label_column: str) -> dict:
        """
        Validate multi-label data format and provide statistics
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'sample_labels': []
        }
        
        if label_column not in df.columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Label column '{label_column}' not found in data")
            return validation_results
        
        all_labels = []
        empty_rows = 0
        invalid_rows = 0
        
        for idx, row in df.iterrows():
            try:
                labels = DataLoader.parse_multilabel_string(row[label_column])
                if not labels:
                    empty_rows += 1
                else:
                    all_labels.extend(labels)
                    if idx < 5:  # Store first 5 as samples
                        validation_results['sample_labels'].append(labels)
            except Exception as e:
                invalid_rows += 1
                if invalid_rows <= 3:  # Only show first 3 errors
                    validation_results['errors'].append(f"Row {idx}: Invalid label format - {str(e)}")
        
        # Statistics
        if all_labels:
            label_counts = pd.Series(all_labels).value_counts()
            validation_results['statistics'] = {
                'total_rows': len(df),
                'rows_with_labels': len(df) - empty_rows,
                'empty_rows': empty_rows,
                'invalid_rows': invalid_rows,
                'unique_labels': len(label_counts),
                'total_label_instances': len(all_labels),
                'avg_labels_per_row': len(all_labels) / max(1, len(df) - empty_rows),
                'most_common_labels': label_counts.head(10).to_dict()
            }
        
        # Warnings
        if empty_rows > 0:
            validation_results['warnings'].append(f"{empty_rows} rows have no labels")
        
        if invalid_rows > 0:
            validation_results['warnings'].append(f"{invalid_rows} rows have invalid label format")
        
        return validation_results

    @staticmethod
    def get_label_distribution(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """
        Get label distribution

        Args:
            df: Input DataFrame
            label_column: Name of the label column

        Returns:
            DataFrame with label counts
        """
        label_counts = df[label_column].value_counts().reset_index()
        label_counts.columns = [label_column, 'Count']
        return label_counts

    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
        """
        Create a download link for a DataFrame

        Args:
            df: DataFrame to download
            filename: Name of the file to download
            link_text: Text to display for the download link

        Returns:
            HTML string with download link
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'

    @staticmethod
    def get_sample_data(classification_type: str = "Multi-Class") -> pd.DataFrame:
        """
        Create sample data for demonstration

        Args:
            classification_type: Type of classification ('Multi-Class' or 'Multi-Label')

        Returns:
            DataFrame with sample data
        """
        if classification_type == "Multi-Label":
            # Multi-label sample data
            sample_data = {
                'text': [
                    'I loved this action-packed superhero movie!',
                    'A romantic comedy that made me laugh and cry.',
                    'This horror film was terrifying and suspenseful.',
                    'An educational documentary about nature.',
                    'A thrilling sports drama about football.',
                    'A family-friendly animated adventure movie.'
                ],
                'labels': [
                    'Action,Superhero,Adventure',
                    'Romance,Comedy',
                    'Horror,Thriller',
                    'Documentary,Educational',
                    'Sports,Drama',
                    'Animation,Family,Adventure'
                ]
            }
        else:
            # Multi-class sample data (original)
            sample_data = {
                'text': [
                    'I loved this movie, it was fantastic!',
                    'The worst film I have ever seen.',
                    'A mediocre experience, neither good nor bad.',
                    'Absolutely brilliant acting and screenplay.'
                ],
                'sentiment': ['positive', 'negative', 'neutral', 'positive']
            }
        
        return pd.DataFrame(sample_data)