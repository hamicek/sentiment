# Sentiment Analysis Tool

A Python tool for analyzing sentiment in product descriptions using the Flair NLP library. The tool processes CSV files containing product data and identifies the most positive and negative products based on sentiment analysis.

## Features

- Sentiment analysis using Flair's pre-trained English sentiment model
- Identifies products with the most positive and negative sentiment
- Word frequency analysis with configurable stop words
- Robust error handling for file operations and data processing
- HTML tag cleaning from product descriptions

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment:
```bash
virtualenv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the sentiment analysis on your dataset:
```bash
python3 main.py
```

The tool will:
1. Load products from `dataset-gymbeam-product-descriptions-eng.csv`
2. Analyze sentiment for each product description
3. Display the most positive and negative products
4. Show the top 10 most frequently used words

## Input Format

The CSV file should have two columns:
- `name`: Product name
- `description`: Product description (HTML tags will be automatically removed)

Example:
```csv
"name","description"
"Product A","<p>This is a <strong>great</strong> product!</p>"
"Product B","<p>Not recommended.</p>"
```

## Output

The program displays:
- **Most positive product**: Product with the highest positive sentiment confidence
- **Most negative product**: Product with the highest negative sentiment confidence
- **Most used words**: Top 10 most frequently occurring words (excluding common stop words)

Example output:
```
max positive:
Caffeine 90 tbl - GymBeam
Caffeine is made by caffeine tablets with up to 200 mg of caffeine...
0.9987

max negative:
Product XYZ
Description here...
0.8543

Most used words:
 - protein - 156
 - support - 89
 - training - 67
 ...
```

## Project Structure

- `main.py` - Main application code with sentiment analysis logic
- `requirements.txt` - Python dependencies
- `dataset-gymbeam-product-descriptions-eng.csv` - Sample dataset
- `.gitignore` - Git ignore rules

## Error Handling

The tool handles various error scenarios:
- Missing or inaccessible CSV files
- Invalid CSV format
- Sentiment analysis failures
- Empty datasets

Errors are reported to stderr with descriptive messages.

## Testing

Run the unit tests to verify functionality:
```bash
python3 -m unittest test_main.py
```

The test suite covers:
- Product initialization
- Sentiment analysis statistics
- Word frequency counting
- CSV loading and validation
- HTML tag removal
- Error handling
