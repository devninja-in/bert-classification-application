import unittest
import torch
from models.model import BERTClassifier
from sklearn.preprocessing import LabelEncoder


class TestBERTClassifier(unittest.TestCase):
    """Test cases for BERTClassifier"""

    def setUp(self):
        """Set up test environment"""
        self.num_labels = 3
        self.classifier = BERTClassifier(
            model_name="distilbert-base-uncased",
            num_labels=self.num_labels,
            max_length=64
        )

        # Create a simple label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = ["negative", "neutral", "positive"]

        # Set it in the classifier
        self.classifier.label_encoder = self.label_encoder

    def test_tokenizer_loading(self):
        """Test that tokenizer loads correctly"""
        self.classifier.load_tokenizer()
        self.assertIsNotNone(self.classifier.tokenizer)

    def test_model_loading(self):
        """Test that model loads correctly"""
        self.classifier.load_model()
        self.assertIsNotNone(self.classifier.model)

        # Check that the model has the right number of labels
        config = self.classifier.model.config
        self.assertEqual(config.num_labels, self.num_labels)

    def test_tokenization(self):
        """Test text tokenization"""
        self.classifier.load_tokenizer()

        texts = ["This is a test sentence", "Another test sentence"]
        encodings = self.classifier.tokenize(texts, max_length=64)

        # Check that the encodings have the right shape
        self.assertEqual(encodings['input_ids'].shape[0], 2)  # 2 sentences
        self.assertLessEqual(encodings['input_ids'].shape[1], 64)  # max length

        # Check that we have attention masks
        self.assertIn('attention_mask', encodings)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_prediction(self):
        """Test prediction (only on CUDA)"""
        self.classifier.load_tokenizer()
        self.classifier.load_model()

        texts = ["This is amazing!"]

        # This should not raise an exception
        labels, probs = self.classifier.predict(texts)

        # Check that we got one label
        self.assertEqual(len(labels), 1)

        # Check that the label is one of our classes
        self.assertIn(labels[0], self.label_encoder.classes_)


if __name__ == '__main__':
    unittest.main()