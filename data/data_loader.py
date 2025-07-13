import pandas as pd
from typing import Optional
import base64

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
    def preprocess_data(df: pd.DataFrame, label_column: str, min_samples: int = 5) -> pd.DataFrame:
        """
        Preprocess data for training

        Args:
            df: Input DataFrame
            label_column: Name of the label column
            min_samples: Minimum number of samples per class

        Returns:
            Preprocessed DataFrame
        """
        # Remove entries with less than min_samples in a class
        filtered_df = df[df.groupby(label_column)[label_column].transform('count') >= min_samples]

        logger.info(f"Preprocessing data: {len(df)} -> {len(filtered_df)} rows after filtering")
        return filtered_df

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
    def get_sample_data() -> pd.DataFrame:
        """
        Create sample data for demonstration

        Returns:
            DataFrame with sample data
        """
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