import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

from utils.logger import logger
from config.config import MODELS_DIR, MULTI_LABEL_THRESHOLD


class BERTClassifier:
    """BERT-based text classification model"""

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            num_labels: int = None,
            max_length: int = 128
    ):
        """
        Initialize BERT classifier

        Args:
            model_name: Name of the pretrained model
            num_labels: Number of labels (classes)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

        logger.info(f"Initializing BERTClassifier with {model_name}")

    def load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_model(self):
        """Load model"""
        if self.num_labels is None:
            raise ValueError("num_labels must be set before loading model")

        logger.info(f"Loading model: {self.model_name} with {self.num_labels} labels")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

    def tokenize(self, texts: List[str], padding: str = 'max_length', truncation: bool = True) -> Dict[
        str, torch.Tensor]:
        """
        Tokenize input texts

        Args:
            texts: List of input texts
            padding: Padding strategy
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        return self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict(self, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Make predictions on texts

        Args:
            texts: List of input texts

        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            raise ValueError("Model, tokenizer, and label_encoder must be loaded before prediction")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        all_labels = []
        all_probs = []

        for text in texts:
            # Tokenize input text
            inputs = self.tokenize([text])

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            # Get predicted label and probabilities
            predicted_label = self.label_encoder.inverse_transform(preds)[0]
            probabilities = probs.cpu().numpy()[0].tolist()

            all_labels.append(predicted_label)
            all_probs.append(probabilities)

        return all_labels, all_probs

    def save(self, model_name: str) -> Tuple[str, str]:
        """
        Save model and components

        Args:
            model_name: Name to save model under

        Returns:
            Tuple of (model_path, components_path)
        """
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            raise ValueError("Model, tokenizer, and label_encoder must be loaded before saving")

        os.makedirs(MODELS_DIR, exist_ok=True)

        save_path = os.path.join(MODELS_DIR, model_name)
        model_path = f"{save_path}_model.pth"
        components_path = f"{save_path}_components.pkl"

        # Save model state dictionary
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer and label encoder
        with open(components_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'max_length': self.max_length,
                'model_name': self.model_name
            }, f)

        logger.info(f"Model saved to {model_path} and {components_path}")

        return model_path, components_path

    @classmethod
    def load(cls, model_path: str, components_path: str) -> "BERTClassifier":
        """
        Load saved model and components

        Args:
            model_path: Path to model file
            components_path: Path to components file

        Returns:
            Loaded BERTClassifier instance
        """
        # Load components
        with open(components_path, 'rb') as f:
            components = pickle.load(f)

        tokenizer = components['tokenizer']
        label_encoder = components['label_encoder']
        max_length = components.get('max_length', 128)
        model_name = components.get('model_name', 'distilbert-base-uncased')

        num_labels = len(label_encoder.classes_)

        # Create classifier instance
        classifier = cls(model_name=model_name, num_labels=num_labels, max_length=max_length)
        classifier.tokenizer = tokenizer
        classifier.label_encoder = label_encoder

        # Initialize model with correct number of labels
        classifier.load_model()

        # Load model state
        classifier.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )

        logger.info(f"Model loaded from {model_path} and {components_path}")

        return classifier


class BERTMultiLabelModel(nn.Module):
    """PyTorch model for multi-label BERT classification"""
    
    def __init__(self, model_name: str, num_labels: int):
        super(BERTMultiLabelModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different BERT model outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # Use pooler_output if available
            pooled_output = outputs.pooler_output
        else:
            # Manually pool from last_hidden_state using [CLS] token
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0]  # Use [CLS] token (first token)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTMultiLabelClassifier:
    """BERT-based multi-label text classification model"""

    def __init__(
            self,
            model_name: str = "distilbert-base-uncased",
            num_labels: int = None,
            max_length: int = 128,
            threshold: float = MULTI_LABEL_THRESHOLD
    ):
        """
        Initialize BERT multi-label classifier

        Args:
            model_name: Name of the pretrained model
            num_labels: Number of labels (classes)
            max_length: Maximum sequence length
            threshold: Threshold for multi-label prediction
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.label_binarizer = None

        logger.info(f"Initializing BERTMultiLabelClassifier with {model_name}")

    def load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_model(self):
        """Load model"""
        if self.num_labels is None:
            raise ValueError("num_labels must be set before loading model")

        logger.info(f"Loading multi-label model: {self.model_name} with {self.num_labels} labels")
        self.model = BERTMultiLabelModel(self.model_name, self.num_labels)

    def tokenize(self, texts: List[str], padding: str = 'max_length', truncation: bool = True) -> Dict[
        str, torch.Tensor]:
        """
        Tokenize input texts

        Args:
            texts: List of input texts
            padding: Padding strategy
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        return self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict(self, texts: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Make multi-label predictions on texts

        Args:
            texts: List of input texts

        Returns:
            Tuple of (predicted_labels_list, probabilities_list)
        """
        if self.model is None or self.tokenizer is None or self.label_binarizer is None:
            raise ValueError("Model, tokenizer, and label_binarizer must be loaded before prediction")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        all_labels = []
        all_probs = []

        for text in texts:
            # Tokenize input text
            inputs = self.tokenize([text])

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Make prediction
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits)
                predictions = (probs > self.threshold).cpu().numpy()

            # Get predicted labels and probabilities
            predicted_labels = self.label_binarizer.inverse_transform(predictions)[0]
            probabilities = probs.cpu().numpy()[0].tolist()

            # Convert to list if single label
            if isinstance(predicted_labels, str):
                predicted_labels = [predicted_labels]
            elif not isinstance(predicted_labels, (list, tuple)):
                predicted_labels = list(predicted_labels)

            all_labels.append(predicted_labels)
            all_probs.append(probabilities)

        return all_labels, all_probs

    def save(self, model_name: str) -> Tuple[str, str]:
        """
        Save model and components

        Args:
            model_name: Name to save model under

        Returns:
            Tuple of (model_path, components_path)
        """
        if self.model is None or self.tokenizer is None or self.label_binarizer is None:
            raise ValueError("Model, tokenizer, and label_binarizer must be loaded before saving")

        os.makedirs(MODELS_DIR, exist_ok=True)

        save_path = os.path.join(MODELS_DIR, model_name)
        model_path = f"{save_path}_multilabel_model.pth"
        components_path = f"{save_path}_multilabel_components.pkl"

        # Save model state dictionary
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer and label binarizer
        with open(components_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_binarizer': self.label_binarizer,
                'max_length': self.max_length,
                'model_name': self.model_name,
                'threshold': self.threshold
            }, f)

        logger.info(f"Multi-label model saved to {model_path} and {components_path}")

        return model_path, components_path

    @classmethod
    def load(cls, model_path: str, components_path: str) -> "BERTMultiLabelClassifier":
        """
        Load saved multi-label model and components

        Args:
            model_path: Path to model file
            components_path: Path to components file

        Returns:
            Loaded BERTMultiLabelClassifier instance
        """
        # Load components
        with open(components_path, 'rb') as f:
            components = pickle.load(f)

        tokenizer = components['tokenizer']
        label_binarizer = components['label_binarizer']
        max_length = components.get('max_length', 128)
        model_name = components.get('model_name', 'distilbert-base-uncased')
        threshold = components.get('threshold', MULTI_LABEL_THRESHOLD)

        num_labels = len(label_binarizer.classes_)

        # Create classifier instance
        classifier = cls(model_name=model_name, num_labels=num_labels, max_length=max_length, threshold=threshold)
        classifier.tokenizer = tokenizer
        classifier.label_binarizer = label_binarizer

        # Initialize and load model
        classifier.load_model()
        classifier.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )

        logger.info(f"Multi-label model loaded from {model_path} and {components_path}")

        return classifier