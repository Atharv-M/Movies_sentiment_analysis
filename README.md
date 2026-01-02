# Movie Sentiment Analysis

## Overview
This project performs sentiment analysis on movie reviews using the IMDB dataset. The goal is to classify reviews as positive or negative based on their content.

## Dataset
The dataset used is the [Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 reviews split evenly into training and test sets, with both positive and negative labels.

- Data location: `aclImdb/`
- Subfolders:
  - `train/` and `test/` contain labeled data
  - `pos/` and `neg/` contain positive and negative reviews
  - `unsup/` contains unlabeled data (for unsupervised learning)

## Project Structure
- `Movies_Sentiment_Analysis.ipynb`: Main Jupyter notebook for data processing, model training, and evaluation
- `aclImdb/`: Dataset folder
- `README.md`: Project documentation
- `LICENSE`: License information

## How to Run
1. Open `Movies_Sentiment_Analysis.ipynb` in Jupyter Notebook or VS Code.
2. Follow the notebook cells to:
   - Load and preprocess the data
   - Train a sentiment analysis model (e.g., using logistic regression, SVM, or deep learning)
   - Evaluate the model on the test set

## Requirements
- Python 3.7+
- Jupyter Notebook
- Common libraries: numpy, pandas, scikit-learn, matplotlib, nltk, etc.

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Results
The notebook will display accuracy, confusion matrix, and example predictions.

## License
This project is licensed under the terms of the LICENSE file in this repository.
