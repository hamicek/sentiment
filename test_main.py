"""Unit tests for sentiment analysis tool."""
import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from models import Product
from statistics import Statistics
from data_loader import DataLoader


class TestProduct(unittest.TestCase):
    """Test cases for the Product class."""

    def test_product_initialization(self):
        """Test that Product initializes with correct attributes."""
        product = Product("Test Product", "Test description")
        self.assertEqual(product.name, "Test Product")
        self.assertEqual(product.description, "Test description")
        self.assertEqual(product.sentiment_value, '')
        self.assertEqual(product.confidence, 0.0)
        self.assertEqual(product.tagged_string, '')


class TestStatistics(unittest.TestCase):
    """Test cases for the Statistics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.products = [
            Product("Product 1", "Great product"),
            Product("Product 2", "Bad product"),
            Product("Product 3", "Amazing product"),
        ]
        # Set up sentiment values
        self.products[0].sentiment_value = 'POSITIVE'
        self.products[0].confidence = 0.95
        self.products[0].tagged_string = 'Great product'

        self.products[1].sentiment_value = 'NEGATIVE'
        self.products[1].confidence = 0.88
        self.products[1].tagged_string = 'Bad product'

        self.products[2].sentiment_value = 'POSITIVE'
        self.products[2].confidence = 0.92
        self.products[2].tagged_string = 'Amazing product'

        self.stats = Statistics(self.products)

    def test_most_positive(self):
        """Test finding the most positive product."""
        result = self.stats.most_positive()
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Product 1")
        self.assertEqual(result.confidence, 0.95)

    def test_most_negative(self):
        """Test finding the most negative product."""
        result = self.stats.most_negative()
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Product 2")
        self.assertEqual(result.confidence, 0.88)

    def test_most_positive_no_positive_products(self):
        """Test most_positive returns None when no positive products exist."""
        products = [Product("Test", "Description")]
        products[0].sentiment_value = 'NEGATIVE'
        products[0].confidence = 0.9
        stats = Statistics(products)
        result = stats.most_positive()
        self.assertIsNone(result)

    def test_most_negative_no_negative_products(self):
        """Test most_negative returns None when no negative products exist."""
        products = [Product("Test", "Description")]
        products[0].sentiment_value = 'POSITIVE'
        products[0].confidence = 0.9
        stats = Statistics(products)
        result = stats.most_negative()
        self.assertIsNone(result)

    def test_words_statistics(self):
        """Test word counting functionality."""
        result = self.stats._words_statistics()
        self.assertIsInstance(result, dict)
        self.assertIn('great', result)
        self.assertIn('bad', result)
        self.assertIn('amazing', result)
        self.assertIn('product', result)
        self.assertEqual(result['product'], 3)

    def test_most_used_words(self):
        """Test finding most frequently used words."""
        result = self.stats.most_used_words(cnt=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # 'product' should be the most common word (appears 3 times)
        self.assertEqual(result[0][0], 'product')
        self.assertEqual(result[0][1], 3)

    def test_most_used_words_with_stop_words(self):
        """Test that stop words are excluded from word statistics."""
        products = [Product("Test", "and but or")]
        products[0].tagged_string = "and but or"
        stats = Statistics(products)
        result = stats._words_statistics()
        self.assertNotIn('and', result)
        self.assertNotIn('but', result)
        self.assertNotIn('or', result)


class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""

    def test_map_to_products_html_removal(self):
        """Test that HTML tags are removed from descriptions."""
        loader = DataLoader("test.csv")
        row = ["Product", "<p>Description with <strong>HTML</strong></p>"]
        product = loader._map_to_products(row)
        self.assertEqual(product.name, "Product")
        self.assertNotIn('<p>', product.description)
        self.assertNotIn('<strong>', product.description)
        self.assertNotIn('</strong>', product.description)
        self.assertNotIn('</p>', product.description)

    def test_map_to_products_invalid_row(self):
        """Test that ValueError is raised for invalid row length."""
        loader = DataLoader("test.csv")
        with self.assertRaises(ValueError):
            loader._map_to_products(["Only one column"])
        with self.assertRaises(ValueError):
            loader._map_to_products(["Col1", "Col2", "Col3"])

    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as f:
            f.write('"name","description"\n')
            f.write('"Product A","<p>Description A</p>"\n')
            f.write('"Product B","<p>Description B</p>"\n')
            temp_file = f.name

        try:
            loader = DataLoader(temp_file)
            loader.load()
            self.assertEqual(len(loader.products), 2)
            self.assertEqual(loader.products[0].name, "Product A")
            self.assertEqual(loader.products[1].name, "Product B")
        finally:
            os.unlink(temp_file)

    def test_load_file_not_found(self):
        """Test that SystemExit is raised when file doesn't exist."""
        loader = DataLoader("nonexistent_file.csv")
        with self.assertRaises(SystemExit):
            loader.load()

    def test_products_property(self):
        """Test that products property returns the internal products list."""
        loader = DataLoader("test.csv")
        self.assertIsInstance(loader.products, list)
        self.assertEqual(len(loader.products), 0)


if __name__ == '__main__':
    unittest.main()
