import numpy as np
import torch
from typing import Tuple, Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import pandas as pd

from models.model import BERTClassifier
from utils.logger import logger


def train_model(
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        test_size: float = 0.2,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        max_length: int = 128,
        model_name: str = "distilbert-base-uncased",
        progress_bar=None,
        status_text=None
) -> Tuple[BERTClassifier, Dict[str, Any], np.ndarray, List[str]]:
    """
    Train BERT model for text classification

    Args:
        df: DataFrame with text and labels
        text_column: Name of text column
        label_column: Name of label column
        test_size: Proportion of test data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        model_name: Name of pretrained model
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text

    Returns:
        Tuple of (classifier, classification_report, confusion_matrix, class_names)
    """
    logger.info(f"Starting model training with {len(df)} samples")

    # Prepare data
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)
    logger.info(f"Found {num_labels} unique labels: {label_encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
    )
    logger.info(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    # Initialize classifier
    classifier = BERTClassifier(model_name=model_name, num_labels=num_labels, max_length=max_length)
    classifier.load_tokenizer()
    classifier.label_encoder = label_encoder

    # Tokenize data
    logger.info("Tokenizing training data")
    train_encodings = classifier.tokenize(X_train)

    logger.info("Tokenizing test data")
    test_encodings = classifier.tokenize(X_test)

    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(y_train)
    )

    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(y_test)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    classifier.load_model()

    # Set up optimizer
    optimizer = AdamW(classifier.model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    classifier.model.to(device)

    if progress_bar is None:
        progress_bar = lambda x: None
    if status_text is None:
        status_text = lambda x: logger.info(x)

    for epoch in range(epochs):
        classifier.model.train()
        total_loss = 0

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = classifier.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Update progress
            progress = (epoch * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader))
            progress_bar(progress)
            status_text(
                f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}"
            )

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} complete - Average loss: {avg_loss:.4f}")

    # Evaluation
    logger.info("Evaluating model on test data")
    classifier.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Generate classification report and confusion matrix
    logger.info("Generating evaluation metrics")
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Training complete - Classification report: {report}")

    return classifier, report, cm, class_names.tolist()