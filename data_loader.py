"""Module for loading product data from CSV files."""
import csv
import logging
import re
import sys
from typing import List

from models import Product

logger = logging.getLogger(__name__)


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
