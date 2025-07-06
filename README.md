# Trigram Language Models

## Overview
This project involves building a trigram language model from scratch in Python. The model implements n-gram extraction, probability estimation, smoothing techniques, and text generation capabilities. The project also includes an application to essay scoring based on perplexity calculations.

## Project Structure

### Core Files
- `trigram_model.py` - Main implementation containing the TrigramModel class and supporting functions
- `brown_train.txt` - Brown corpus training data (American written English from 1950s)
- `brown_test.txt` - Brown corpus test data for perplexity evaluation
- `ets_toefl_data/` - Directory containing TOEFL essay data for classification
  - `train_high.txt` - Training essays scored as "high" skill level
  - `train_low.txt` - Training essays scored as "low" skill level
  - `test_high/` - Test essays (high skill level)
  - `test_low/` - Test essays (low skill level)

### Key Components

#### 1. N-gram Extraction (`get_ngrams`)
- Extracts n-grams from sequences with proper padding
- Handles START and STOP tokens for sentence boundaries
- Supports arbitrary n-gram sizes (implemented for n=1,2,3)

#### 2. TrigramModel Class
- **Initialization**: Builds lexicon and counts n-grams from corpus
- **Counting**: Populates unigram, bigram, and trigram frequency dictionaries
- **Probability Calculation**: Computes raw and smoothed probabilities
- **Text Generation**: Generates sentences using trigram probabilities
- **Evaluation**: Calculates perplexity for model assessment

#### 3. Probability Methods
- `raw_unigram_probability()` - Maximum likelihood estimation for unigrams
- `raw_bigram_probability()` - Maximum likelihood estimation for bigrams
- `raw_trigram_probability()` - Maximum likelihood estimation for trigrams
- `smoothed_trigram_probability()` - Linear interpolation smoothing (λ₁=λ₂=λ₃=1/3)

#### 4. Evaluation Methods
- `sentence_logprob()` - Calculates log probability of sentences
- `perplexity()` - Computes perplexity on test corpora
- `essay_scoring_experiment()` - Classifies essays based on perplexity

## Technical Requirements

### Python Version
- **Python 3.6+** is required. The code is not compatible with Python 2.x.

### Required Packages
This project uses only standard Python libraries:
- `collections`
- `math`
- `random`
- `os`
- `sys`

No external packages are required. However, for best results, use a recent Python 3 distribution (e.g., Anaconda or Miniconda).

### Environment Setup
1. **Install Python 3.6 or higher**
   - [Download Python](https://www.python.org/downloads/)
   - Or use [Anaconda](https://www.anaconda.com/products/distribution)
2. **Clone or download this repository**
3. **Ensure all data files are present**
   - `brown_train.txt`, `brown_test.txt`, and the `ets_toefl_data/` directory must be in the same directory as `trigram_model.py`.

### Running the Code

#### Training and Evaluating the Model
You can run the model interactively or by adding a `main` section to `trigram_model.py`.

**Example: Calculate Perplexity**
```bash
python3
```
```python
from trigram_model import TrigramModel, corpus_reader
model = TrigramModel('brown_train.txt')
test_corpus = corpus_reader('brown_test.txt', model.lexicon)
print("Perplexity:", model.perplexity(test_corpus))
```

**Example: Generate a Sentence**
```python
sentence = model.generate_sentence()
print(' '.join(sentence))
```

**Example: Essay Classification**
```python
from trigram_model import essay_scoring_experiment
accuracy = essay_scoring_experiment(
    'ets_toefl_data/train_high.txt',
    'ets_toefl_data/train_low.txt',
    'ets_toefl_data/test_high/',
    'ets_toefl_data/test_low/'
)
print(f"Classification accuracy: {accuracy:.2%}")
```

#### Troubleshooting
- **FileNotFoundError**: Ensure all data files are in the correct directory.
- **UnicodeDecodeError**: If you encounter encoding issues, open files with `encoding='utf-8'`.
- **Memory Issues**: For very large corpora, ensure your system has at least 2GB RAM.
- **Python Version Errors**: Confirm you are using Python 3.6 or higher (`python --version`).

## Implementation Details

### Vocabulary Handling
- Words appearing only once are replaced with `<UNK>` token
- Special tokens: `START`, `STOP`, `UNK`, `ROOT`, `NULL`
- Lexicon built from training data automatically

### Smoothing Strategy
- Linear interpolation combines unigram, bigram, and trigram probabilities
- Equal weights (1/3) for all three n-gram levels
- Handles unseen n-grams gracefully

### Text Generation Process
1. Initialize with `("START", "START")`
2. Sample next word from trigram distribution
3. Update context window
4. Continue until `STOP` token or maximum length

## Expected Performance
- **Brown Corpus Perplexity**: < 400 on test set
- **Essay Classification Accuracy**: > 80%
- **Training Data Perplexity**: Significantly lower than test (overfitting expected)

## File Format Notes
- Training files: One sentence per line, tokenized
- TOEFL data: Specialized format for essay scoring
- All text preprocessing is handled automatically

## Key Features
- Handles arbitrary vocabulary sizes
- Robust to unseen words and contexts
- Efficient probability computation on demand
- Supports both intrinsic (perplexity) and extrinsic (classification) evaluation

