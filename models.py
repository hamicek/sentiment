"""Data models for sentiment analysis."""


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
