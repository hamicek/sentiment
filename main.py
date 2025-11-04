from flair.models import TextClassifier
from flair.data import Sentence
from operator import attrgetter
import csv
import re
import sys


DATA_FILE_NAME = 'dataset-gymbeam-product-descriptions-eng.csv'


class Product():
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.sentiment_value = ''
        self.confidence = 0
        self.tagged_string = ''


class Products(list):
    pass


class Statistics():
    ONE_WORD_CONJUNCTIONS = ('and', 'but', 'or', 'for', 'nor', 'yet', 'so', '.', ',', '!', '?')

    def __init__(self, products):
        self.products = products

    def compute_sentiment(self):
        try:
            classifier = TextClassifier.load('en-sentiment')
        except Exception as e:
            print(f"Error: Failed to load sentiment classifier: {e}", file=sys.stderr)
            sys.exit(1)

        for p in self.products:
            try:
                sentence = Sentence(p.description)
                classifier.predict(sentence)
                rating = sentence.to_dict()
                p.sentiment_value = rating['labels'][0]['value']
                p.confidence = rating['labels'][0]['confidence']
                p.tagged_string = sentence.to_tagged_string()
            except Exception as e:
                print(f"Warning: Failed to analyze sentiment for '{p.name}': {e}", file=sys.stderr)
                p.sentiment_value = 'UNKNOWN'
                p.confidence = 0.0
                p.tagged_string = p.description

    def most_positive(self):
        positive_products = [p for p in self.products if p.sentiment_value == 'POSITIVE']
        if not positive_products:
            return None
        return max(positive_products, key=attrgetter("confidence"))

    def most_negative(self):
        negative_products = [p for p in self.products if p.sentiment_value == 'NEGATIVE']
        if not negative_products:
            return None
        return max(negative_products, key=attrgetter("confidence"))

    def most_used_words(self, cnt=10, stop_words=ONE_WORD_CONJUNCTIONS):
        words = self._words_statistics(stop_words)
        words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
        words_items = list(words.items())
        words_items = words_items[-min(cnt,len(words_items)):]
        words_items.reverse()
        return words_items

    def _words_statistics(self, stop_words=ONE_WORD_CONJUNCTIONS):
        word_statistics = {}
        for p in self.products:
            words = [w.lower() for w in p.tagged_string.split() if w.lower() not in stop_words]
            for w in words:
                word_statistics[w] = word_statistics.get(w, 0) + 1
        return word_statistics


class DataLoader():
    def __init__(self, file_name):
        self.file_name = file_name
        self._products = Products()

    def load(self):
        try:
            with open(self.file_name, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)  # Skip header row
                for row_num, row in enumerate(csv_reader, start=2):
                    try:
                        product = self._map_to_products(row)
                        self._products.append(product)
                    except ValueError as e:
                        print(f"Warning: Skipping row {row_num} due to error: {e}", file=sys.stderr)
                        continue
        except FileNotFoundError:
            print(f"Error: File '{self.file_name}' not found.", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: Permission denied reading file '{self.file_name}'.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Unexpected error while reading file: {e}", file=sys.stderr)
            sys.exit(1)

    def _get_products(self):
        return self._products

    def _map_to_products(self, row):
        if len(row) != 2:
            raise ValueError(f"Expected 2 columns, got {len(row)}")
        product, description = row
        cleanr = re.compile('<.*?>')
        description = re.sub(cleanr, '', description)
        return Product(product, description)

    products = property(_get_products)


def print_product(product):
    print(product.name)
    print(product.description)
    #print(product.sentiment_value)
    print(product.confidence)
    #print(product.tagged_string)


if __name__ == "__main__":
    data_loader = DataLoader(DATA_FILE_NAME)
    data_loader.load()

    if not data_loader.products:
        print("Error: No products loaded from file.", file=sys.stderr)
        sys.exit(1)

    statistics = Statistics(data_loader.products)
    statistics.compute_sentiment()

    max_positive = statistics.most_positive()
    max_negative = statistics.most_negative()

    print('max positive:')
    if max_positive:
        print_product(max_positive)
    else:
        print('No positive products found')
    print()
    print('max negative:')
    if max_negative:
        print_product(max_negative)
    else:
        print('No negative products found')
    print()
    print('Most used words:')
    words = statistics.most_used_words()
    for w in words:
        print(' - %s - %s' % (w[0], w[1]))
