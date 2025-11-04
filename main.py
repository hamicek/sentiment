"""Main entry point for the sentiment analysis tool."""
import argparse
import logging
import sys

from models import Product
from data_loader import DataLoader
from statistics import Statistics
from config import DEFAULT_CSV_FILE, DEFAULT_WORD_COUNT

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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
        default=DEFAULT_CSV_FILE,
        help=f'Path to the CSV file with product data (default: {DEFAULT_CSV_FILE})'
    )
    parser.add_argument(
        '--words', '-w',
        type=int,
        default=DEFAULT_WORD_COUNT,
        help=f'Number of most used words to display (default: {DEFAULT_WORD_COUNT})'
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
