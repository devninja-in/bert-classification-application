import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
from typing import Dict, Any

from utils.logger import logger


def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix as a base64 encoded image

    Args:
        cm: Confusion matrix array
        classes: Class names

    Returns:
        Base64 encoded image
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        logger.error(f"Error generating confusion matrix plot: {str(e)}")
        return None


def plot_probability_chart(probabilities: Dict[str, float]):
    """
    Plot probability chart as a base64 encoded image

    Args:
        probabilities: Dictionary of class probabilities

    Returns:
        Base64 encoded image
    """
    try:
        classes = list(probabilities.keys())
        probs = list(probabilities.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, probs)
        plt.ylabel('Probability')
        plt.xlabel('Class')
        plt.title('Prediction Probabilities')
        plt.ylim(0, 1)

        # Add probability values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        logger.error(f"Error generating probability chart: {str(e)}")
        return None


def format_classification_report(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Format classification report as DataFrame

    Args:
        report: Classification report dictionary

    Returns:
        Formatted DataFrame
    """
    try:
        df = pd.DataFrame(report).transpose()

        # Reorder columns if necessary
        if 'precision' in df.columns and 'recall' in df.columns and 'f1-score' in df.columns:
            df = df[['precision', 'recall', 'f1-score', 'support']]

        # Round numeric columns
        for col in df.columns:
            if col != 'support':
                df[col] = df[col].round(3)

        return df
    except Exception as e:
        logger.error(f"Error formatting classification report: {str(e)}")
        return pd.DataFrame(report).transpose()