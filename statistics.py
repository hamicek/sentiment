"""Module for sentiment analysis and word statistics."""
import logging
import sys
from flair.models import TextClassifier
from flair.data import Sentence
from operator import attrgetter
from typing import List, Dict, Tuple, Optional

from models import Product
from config import STOP_WORDS

logger = logging.getLogger(__name__)


class Statistics:
    """Performs sentiment analysis and statistics on a collection of products.

    Attributes:
        ONE_WORD_CONJUNCTIONS: Common stop words to exclude from word statistics
        products: List of Product objects to analyze
    """
    ONE_WORD_CONJUNCTIONS: Tuple[str, ...] = STOP_WORDS

    def __init__(self, products: List[Product]) -> None:
        self.products: List[Product] = products

    def compute_sentiment(self) -> None:
        """Computes sentiment analysis for all products using Flair's pre-trained model.

        Updates each product's sentiment_value, confidence, and tagged_string attributes.
        Exits the program if the sentiment classifier cannot be loaded.
        """
        try:
            logger.info("Loading sentiment classifier...")
            classifier = TextClassifier.load('en-sentiment')
            logger.info("Sentiment classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment classifier: {e}")
            sys.exit(1)

        logger.info(f"Analyzing sentiment for {len(self.products)} products...")
        for p in self.products:
            try:
                sentence = Sentence(p.description)
                classifier.predict(sentence)
                rating = sentence.to_dict()
                p.sentiment_value = rating['labels'][0]['value']
                p.confidence = rating['labels'][0]['confidence']
                p.tagged_string = sentence.to_tagged_string()
            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for '{p.name}': {e}")
                p.sentiment_value = 'UNKNOWN'
                p.confidence = 0.0
                p.tagged_string = p.description
        logger.info("Sentiment analysis completed")

    def most_positive(self) -> Optional[Product]:
        """Finds the product with the highest positive sentiment confidence.

        Returns:
            The product with the highest positive sentiment confidence, or None if no
            positive products are found.
        """
        positive_products = [p for p in self.products if p.sentiment_value == 'POSITIVE']
        if not positive_products:
            return None
        return max(positive_products, key=attrgetter("confidence"))

    def most_negative(self) -> Optional[Product]:
        """Finds the product with the highest negative sentiment confidence.

        Returns:
            The product with the highest negative sentiment confidence, or None if no
            negative products are found.
        """
        negative_products = [p for p in self.products if p.sentiment_value == 'NEGATIVE']
        if not negative_products:
            return None
        return max(negative_products, key=attrgetter("confidence"))

    def most_used_words(self, cnt: int = 10, stop_words: Tuple[str, ...] = ONE_WORD_CONJUNCTIONS) -> List[Tuple[str, int]]:
        """Finds the most frequently used words across all product descriptions.

        Args:
            cnt: Maximum number of words to return (default: 10)
            stop_words: Tuple of words to exclude from counting (default: ONE_WORD_CONJUNCTIONS)

        Returns:
            List of (word, count) tuples sorted by frequency in descending order.
        """
        words = self._words_statistics(stop_words)
        words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
        words_items = list(words.items())
        words_items = words_items[-min(cnt, len(words_items)):]
        words_items.reverse()
        return words_items

    def _words_statistics(self, stop_words: Tuple[str, ...] = ONE_WORD_CONJUNCTIONS) -> Dict[str, int]:
        """Counts word frequency across all products, excluding stop words.

        Args:
            stop_words: Tuple of words to exclude from counting

        Returns:
            Dictionary mapping words to their occurrence count.
        """
        word_statistics: Dict[str, int] = {}
        for p in self.products:
            words = [w.lower() for w in p.tagged_string.split() if w.lower() not in stop_words]
            for w in words:
                word_statistics[w] = word_statistics.get(w, 0) + 1
        return word_statistics
