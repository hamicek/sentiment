from flair.models import TextClassifier
from flair.data import Sentence
from operator import attrgetter
import csv
import re


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
        classifier = TextClassifier.load('en-sentiment')
        for p in self.products:
            sentence = Sentence(p.description)
            classifier.predict(sentence)
            rating = sentence.to_dict()
            p.sentiment_value = rating['labels'][0]['value']
            p.confidence = rating['labels'][0]['confidence']
            p.tagged_string = sentence.to_tagged_string()

    def most_positive(self):
        return max([p for p in self.products if p.sentiment_value == 'POSITIVE'], 
            key=attrgetter("confidence"))

    def most_negative(self):
        return max([p for p in self.products if p.sentiment_value == 'NEGATIVE'], 
            key=attrgetter("confidence"))

    def most_used_words(self, cnt=10, stop_words=ONE_WORD_CONJUNCTIONS):
        words = self._words_statistics(stop_words)
        words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
        words_items = list(words.items())
        words_items.reverse()
        return words_items[:min(cnt,len(words_items))]

    def _words_statistics(self, stop_words=ONE_WORD_CONJUNCTIONS):
        word_statistics = {}
        for p in self.products:
            words = [w.lower() for w in p.tagged_string.split() if w.lower() not in stop_words]
            for w in words:
                if not w in word_statistics:
                    word_statistics[w] = words.count(w)
                else:
                    word_statistics[w] = word_statistics[w] + words.count(w)
        return word_statistics


class DataLoader():
    def __init__(self, file_name):
        self.file_name = file_name
        self._products = Products()

    def load(self):
        with open(self.file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                product = self._map_to_products(row)
                self._products.append(product)

    def _get_products(self):
        return self._products

    def _map_to_products(self, row):
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

    statistics = Statistics(data_loader.products)
    statistics.compute_sentiment()

    max_positive = statistics.most_positive()
    max_negative = statistics.most_negative()

    print('max positive:')
    print_product(max_positive)
    print()
    print('max negative:')
    print_product(max_negative)
    print()
    print('Most used words:')
    words = statistics.most_used_words()
    for w in words:
        print(' - %s - %s' % (w[0], w[1]))
