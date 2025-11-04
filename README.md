# Sentiment Analysis Tool

A Python tool for analyzing sentiment in product descriptions using the Flair NLP library. The tool processes CSV files containing product data and identifies the most positive and negative products based on sentiment analysis.

## Features

- **Sentiment analysis** using Flair's pre-trained English sentiment model
- **Most positive/negative detection** - identifies products with extreme sentiment
- **Word frequency analysis** with configurable stop words and output count
- **Command-line interface** with flexible arguments (file path, word count, verbosity)
- **Structured logging** with timestamps and severity levels
- **Robust error handling** for file operations and data processing
- **HTML tag cleaning** from product descriptions
- **Type hints** for better code maintainability
- **Comprehensive unit tests** for reliability

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

### Basic Usage

Run the sentiment analysis with default settings:
```bash
python3 main.py
```

### Command-Line Options

The tool supports several command-line arguments:

```bash
# Analyze a different CSV file
python3 main.py --file my_products.csv

# Show top 20 most used words instead of default 10
python3 main.py --words 20

# Enable verbose logging for debugging
python3 main.py --verbose

# Combine options
python3 main.py --file products.csv --words 15 --verbose
```

**Available arguments:**
- `--file` / `-f`: Path to the CSV file (default: `dataset-gymbeam-product-descriptions-eng.csv`)
- `--words` / `-w`: Number of most used words to display (default: 10)
- `--verbose` / `-v`: Enable verbose logging (DEBUG level)
- `--help` / `-h`: Show help message with all options

### How It Works

The tool will:
1. Load products from the specified CSV file
2. Analyze sentiment for each product description using Flair's pre-trained model
3. Display the most positive and negative products
4. Show the top N most frequently used words (configurable)

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
- **Most used words**: Top N most frequently occurring words (configurable, default: 10)

The tool also provides informative logging messages during execution to track progress.

Example output:
```
2025-11-04 18:30:15 - INFO - Starting sentiment analysis tool
2025-11-04 18:30:15 - INFO - Loading products from 'dataset-gymbeam-product-descriptions-eng.csv'...
2025-11-04 18:30:15 - INFO - Successfully loaded 100 products
2025-11-04 18:30:15 - INFO - Loading sentiment classifier...
2025-11-04 18:30:18 - INFO - Sentiment classifier loaded successfully
2025-11-04 18:30:18 - INFO - Analyzing sentiment for 100 products...
2025-11-04 18:30:25 - INFO - Sentiment analysis completed

============================================================
SENTIMENT ANALYSIS RESULTS
============================================================

Most Positive Product:
----------------------------------------
Caffeine 90 tbl - GymBeam
Caffeine is made by caffeine tablets with up to 200 mg of caffeine...
0.9987

----------------------------------------
Most Negative Product:
----------------------------------------
Product XYZ
Description here...
0.8543

----------------------------------------
Top 10 Most Used Words:
----------------------------------------
 1. protein              - 156 occurrences
 2. support              - 89 occurrences
 3. training             - 67 occurrences
 ...

============================================================
2025-11-04 18:30:25 - INFO - Analysis complete
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
