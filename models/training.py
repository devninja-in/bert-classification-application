import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, hamming_loss, jaccard_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import pandas as pd
import ast

from models.model import BERTClassifier, BERTMultiLabelClassifier
from utils.logger import logger
from config.config import DEFAULT_CLASSIFICATION_TYPE


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
        classification_type: str = DEFAULT_CLASSIFICATION_TYPE,
        progress_bar=None,
        status_text=None
) -> Tuple[Union[BERTClassifier, BERTMultiLabelClassifier], Dict[str, Any], np.ndarray, List[str]]:
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
        classification_type: Type of classification ('Multi-Class' or 'Multi-Label')
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text

    Returns:
        Tuple of (classifier, classification_report, confusion_matrix, class_names)
    """
    logger.info(f"Starting {classification_type} model training with {len(df)} samples")

    # Prepare data
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    if classification_type == "Multi-Label":
        return train_multilabel_model(
            texts, labels, test_size, epochs, batch_size, learning_rate, 
            max_length, model_name, progress_bar, status_text
        )
    else:
        # Multi-Class training (original logic)
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

        # Ensure model_name is a valid HuggingFace identifier  
        valid_model_names = ["distilbert-base-uncased", "bert-base-uncased"]
        if model_name not in valid_model_names:
            logger.warning(f"Invalid model_name '{model_name}', defaulting to 'distilbert-base-uncased'")
            model_name = "distilbert-base-uncased"
        
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


def parse_multilabel_string(label_str: str) -> List[str]:
    """
    Parse multi-label string into list of labels
    Supports formats: "label1,label2", "['label1','label2']", etc.
    """
    if pd.isna(label_str) or label_str == "":
        return []
    
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


def train_multilabel_model(
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        max_length: int = 128,
        model_name: str = "distilbert-base-uncased",
        progress_bar=None,
        status_text=None
) -> Tuple[BERTMultiLabelClassifier, Dict[str, Any], np.ndarray, List[str]]:
    """
    Train BERT model for multi-label text classification
    """
    logger.info("Processing multi-label data")
    
    # Parse multi-label strings
    parsed_labels = [parse_multilabel_string(label) for label in labels]
    
    # Remove empty label entries
    valid_indices = [i for i, lbls in enumerate(parsed_labels) if lbls]
    texts = [texts[i] for i in valid_indices]
    parsed_labels = [parsed_labels[i] for i in valid_indices]
    
    # Encode labels with MultiLabelBinarizer
    label_binarizer = MultiLabelBinarizer()
    encoded_labels = label_binarizer.fit_transform(parsed_labels)
    num_labels = len(label_binarizer.classes_)
    logger.info(f"Found {num_labels} unique labels: {label_binarizer.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=test_size, random_state=42
    )
    logger.info(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Ensure model_name is a valid HuggingFace identifier
    valid_model_names = ["distilbert-base-uncased", "bert-base-uncased"]
    if model_name not in valid_model_names:
        logger.warning(f"Invalid model_name '{model_name}', defaulting to 'distilbert-base-uncased'")
        model_name = "distilbert-base-uncased"
    
    # Initialize classifier
    classifier = BERTMultiLabelClassifier(model_name=model_name, num_labels=num_labels, max_length=max_length)
    classifier.load_tokenizer()
    classifier.load_model()
    classifier.label_binarizer = label_binarizer
    
    # Tokenize data
    logger.info("Tokenizing training data")
    train_encodings = classifier.tokenize(X_train)
    
    logger.info("Tokenizing test data")
    test_encodings = classifier.tokenize(X_test)
    
    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(y_train, dtype=torch.float)
    )
    
    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(y_test, dtype=torch.float)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set up optimizer and loss function
    optimizer = AdamW(classifier.model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
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
            
            logits = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
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
    logger.info("Evaluating multi-label model on test data")
    classifier.model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            logits = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > classifier.threshold).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate multilabel metrics
    logger.info("Generating multi-label evaluation metrics")
    class_names = label_binarizer.classes_
    
    # Multi-label specific metrics
    hamming = hamming_loss(all_labels, all_preds)
    jaccard = jaccard_score(all_labels, all_preds, average='samples')
    
    # Create a report-like dictionary
    report = {
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'accuracy': np.mean(all_labels == all_preds)
    }
    
    # Multi-label confusion matrix (for visualization)
    # Use hamming loss as a simple metric for confusion matrix visualization
    cm = np.array([[hamming, 1-hamming], [1-hamming, hamming]])
    
    logger.info(f"Multi-label training complete - Hamming Loss: {hamming:.4f}, Jaccard Score: {jaccard:.4f}")
    
    return classifier, report, cm, class_names.tolist()