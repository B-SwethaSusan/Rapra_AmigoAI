# Project AmigoAI

This repository contains two main projects:

1. **RAG (Retrieval-Augmented Generation) System** - A system for question answering using document retrieval and language models
2. **Sentiment Analysis Model** - A fine-tuned DistilBERT model for sentiment classification

## Quick Start

### Local Setup

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/B-SwethaSusan/Rapra_AmigoAI
   cd Project_AmigoAI
   pip install -r requirements.txt
   ```

## Table of Contents

* [Prerequisites](#prerequisites)
* [Setup](#setup)
* [Sentiment Analysis Project](#sentiment-analysis-project)

  * [Training](#training-the-model)
  * [Inference](#running-inference)
* [RAG Project](#rag-project)

  * [Setup](#rag-setup)
  * [Running the RAG System](#running-the-rag-system)
* [Project Structure](#project-structure)
* [Model Card](#model-card)

---

## Prerequisites

* Python 3.8+
* pip (Python package manager)
* CUDA-compatible GPU (recommended for faster training)

---

## Setup

1. **Clone the repository:**

   ```bash
   git clone <https://github.com/B-SwethaSusan/Rapra_AmigoAI>
   cd Project_AmigoAI
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   
   venv\Scripts\activate    
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Sentiment Analysis Project

### Training the Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook src/train.ipynb
   ```
   Or open it directly in your preferred Jupyter environment.

2. **Configure the training parameters** in the first cell:
   ```python
   # Configuration
   MODEL_NAME = "distilbert-base-uncased"
   DATA_DIR = "../data/sentiment"
   OUTPUT_DIR = "../submissions/sentiment_test_predictions.csv"
   BATCH_SIZE = 16
   LEARNING_RATE = 2e-5
   NUM_EPOCHS = 5
   ```

3. **Run all cells** to:
   - Load and preprocess the data
   - Initialize the model
   - Train the model
   - Save checkpoints
   - Evaluate performance

### Making Predictions

1. In the same notebook, navigate to the "Inference" section
2. Update the prediction parameters if needed:
   ```python
   MODEL_PATH = "../src/train.ipynb"  
   INPUT_FILE = "../data/sentiment/test.csv"
   OUTPUT_FILE = "../submissions/sentiment_test_predictions.csv"
   ```
3. Run the inference cells to generate predictions

---

## RAG (Retrieval-Augmented Generation) Project

### Setup

1. **Prepare your data files**:
   - Place your document corpus in `data/corpus/docs.jsonl`
   - Place your questions in `data/corpus/questions.json`

### Running the RAG System

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook src/rag_answer.ipynb
   ```
   Or open it directly in your preferred Jupyter environment.

2. **Configure the RAG system** in the first cell:
   ```python
   # Configuration
   DOCS_PATH = "../data/corpus/docs.jsonl"
   QUESTIONS_PATH = "../data/corpus/questions.json"
   OUTPUT_DIR = "../submissions/rag_answers.json"
   MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
   TOP_K = 5 
   ```

3. **Run all cells** to:
   - Load and preprocess documents
   - Initialize the retriever and generator
   - Process questions
   - Generate and save answers

### Output
- The system will save answers to `../submissions/rag_answers.json`
- Intermediate results and retrieved documents are displayed in the notebook

---

## Project Structure

```
Project_AmigoAI/
├── data/                         # All data files
│   ├── corpus/                   # Documents for RAG system
│   │   ├── docs.jsonl            # Document collection (one JSON per line)
│   │   └── questions.json        # Questions to answer with RAG
│   └── sentiment/                # Sentiment analysis datasets
│       ├── train.csv             # Training data (text, label)
│       ├── dev.csv               # Validation data
│       └── test.csv              # Test data (for final evaluation)
│
├── src/                          # Source code
│   ├── rag_answer.ipynb         # RAG system implementation
│   └── train.ipynb              # Sentiment analysis training & inference
│
├── submissions/                  # Generated outputs
│   ├── rag_answers.json         # RAG system predictions
│   └── sentiment_predictions.csv # Sentiment predictions
│
├── .gitignore                   # Git ignore file
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── MODEL_CARD.md              # Model details and evaluation
└── RAG_README.md              # RAG system documentation
```

## Documentation

- [MODEL_CARD.md](MODEL_CARD.md): Detailed information about the sentiment analysis model
- [RAG_README.md](RAG_README.md): Detailed documentation for the RAG system

## Model Card

For detailed information about the sentiment analysis model architecture, performance, and limitations, see [MODEL_CARD.md](MODEL_CARD.md).



