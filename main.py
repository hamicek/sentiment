from flair.models import TextClassifier
from flair.data import Sentence
from operator import attrgetter
from typing import List, Dict, Tuple, Optional
import argparse
import csv
import logging
import re
import sys


DATA_FILE_NAME = 'dataset-gymbeam-product-descriptions-eng.csv'

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Product:
    """Represents a product with its description and sentiment analysis results.

    Attributes:
        name: The name of the product
        description: The product description text
        sentiment_value: The sentiment classification (POSITIVE/NEGATIVE/UNKNOWN)
        confidence: The confidence score of the sentiment prediction (0.0-1.0)
        tagged_string: The text with sentiment tags applied
    """

    def __init__(self, name: str, description: str) -> None:
        self.name: str = name
        self.description: str = description
        self.sentiment_value: str = ''
        self.confidence: float = 0.0
        self.tagged_string: str = ''


class Statistics:
    """Performs sentiment analysis and statistics on a collection of products.

    Attributes:
        ONE_WORD_CONJUNCTIONS: Common stop words to exclude from word statistics
        products: List of Product objects to analyze
    """
    ONE_WORD_CONJUNCTIONS: Tuple[str, ...] = ('and', 'but', 'or', 'for', 'nor', 'yet', 'so', '.', ',', '!', '?')

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


class DataLoader:
    """Loads product data from a CSV file.

    Attributes:
        file_name: Path to the CSV file to load
        _products: Internal list of loaded Product objects
    """
    HTML_TAG_PATTERN = re.compile('<.*?>')

    def __init__(self, file_name: str) -> None:
        self.file_name: str = file_name
        self._products: List[Product] = []

    def load(self) -> None:
        """Loads products from the CSV file.

        Reads a CSV file with 'name' and 'description' columns, skips the header row,
        and creates Product objects. Invalid rows are skipped with a warning.

        Raises:
            SystemExit: If the file is not found, permission is denied, or another error occurs.
        """
        logger.info(f"Loading products from '{self.file_name}'...")
        try:
            with open(self.file_name, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)  # Skip header row
                for row_num, row in enumerate(csv_reader, start=2):
                    try:
                        product = self._map_to_products(row)
                        self._products.append(product)
                    except ValueError as e:
                        logger.warning(f"Skipping row {row_num} due to error: {e}")
                        continue
            logger.info(f"Successfully loaded {len(self._products)} products")
        except FileNotFoundError:
            logger.error(f"File '{self.file_name}' not found")
            sys.exit(1)
        except PermissionError:
            logger.error(f"Permission denied reading file '{self.file_name}'")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error while reading file: {e}")
            sys.exit(1)

    def _get_products(self) -> List[Product]:
        """Returns the list of loaded products."""
        return self._products

    def _map_to_products(self, row: List[str]) -> Product:
        """Converts a CSV row to a Product object.

        Args:
            row: List of CSV values (expected: [name, description])

        Returns:
            A Product object with HTML tags removed from description.

        Raises:
            ValueError: If the row doesn't contain exactly 2 columns.
        """
        if len(row) != 2:
            raise ValueError(f"Expected 2 columns, got {len(row)}")
        product, description = row
        description = re.sub(self.HTML_TAG_PATTERN, '', description)
        return Product(product, description)

    products = property(_get_products)


def print_product(product: Product) -> None:
    """Prints product information to stdout.

    Args:
        product: The Product object to display
    """
    print(product.name)
    print(product.description)
    print(product.confidence)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Sentiment analysis tool for product descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --file my_products.csv
  python main.py --file products.csv --words 20
  python main.py --verbose
        """
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=DATA_FILE_NAME,
        help=f'Path to the CSV file with product data (default: {DATA_FILE_NAME})'
    )
    parser.add_argument(
        '--words', '-w',
        type=int,
        default=10,
        help='Number of most used words to display (default: 10)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run sentiment analysis."""
    args = parse_arguments()

    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    logger.info("Starting sentiment analysis tool")

    # Load data
    data_loader = DataLoader(args.file)
    data_loader.load()

    if not data_loader.products:
        logger.error("No products loaded from file")
        sys.exit(1)

    # Compute sentiment
    statistics = Statistics(data_loader.products)
    statistics.compute_sentiment()

    # Display results
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60 + "\n")

    max_positive = statistics.most_positive()
    max_negative = statistics.most_negative()

    print('Most Positive Product:')
    print('-' * 40)
    if max_positive:
        print_product(max_positive)
    else:
        print('No positive products found')

    print('\n' + '-' * 40)
    print('Most Negative Product:')
    print('-' * 40)
    if max_negative:
        print_product(max_negative)
    else:
        print('No negative products found')

    print('\n' + '-' * 40)
    print(f'Top {args.words} Most Used Words:')
    print('-' * 40)
    words = statistics.most_used_words(cnt=args.words)
    for i, (word, count) in enumerate(words, 1):
        print(f'{i:2d}. {word:20s} - {count} occurrences')

    print("\n" + "="*60)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
