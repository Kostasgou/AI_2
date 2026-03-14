# IMDB Sentiment Analysis Project in Python

A complete Artificial Intelligence and Machine Learning project for **sentiment analysis** on the **IMDB movie reviews dataset**, implemented in **Python**.
The project combines **custom implementations** of classical machine learning algorithms with **library-based baselines** and a **neural network approach** using **PyTorch**.

This repository is designed as an experimental and educational study of text classification, focusing on how different models perform on the task of identifying whether a movie review is **positive** or **negative**.

---

## Overview

The goal of this project is to solve a **binary text classification** problem using the IMDB review dataset.

Each review belongs to one of two sentiment classes:

* **Positive**
* **Negative**

To address this task, the project explores multiple approaches, ranging from traditional probabilistic and linear models to ensemble methods and neural networks.

The repository is structured around several experimental parts, each focusing on a different model family or a comparison between custom implementations and well-known machine learning libraries.

The main approaches included are:

* **Naive Bayes**
* **Logistic Regression**
* **AdaBoost**
* **Simple RNN** with PyTorch

In addition, the project compares several custom implementations against equivalent algorithms from **scikit-learn**, making it both an implementation exercise and a model evaluation study.

---

## Main Objectives

This project was developed to demonstrate and compare different approaches to sentiment classification.

Its main objectives are:

* build custom classifiers for text sentiment analysis
* compare custom implementations with established library models
* evaluate model quality using standard classification metrics
* analyze the impact of training size and feature selection
* explore both classical machine learning and neural network methods
* apply AI and machine learning theory to a realistic NLP dataset

---

## Project Structure

```text
AI_2/
├── AdaBoost.py
├── Adaboost_main.py
├── Adaboost_main_B.py
├── Bayes.py
├── IMDBDDataset.py
├── Logistic_Regression.py
├── ParrtG.py
├── main_bayes.py
├── main_bayes_part2.py
├── main_partA_Logistic.py
├── main_partB_Logistic.py
├── README.md
└── .gitignore
```

---

## Technologies Used

This project combines several tools and libraries commonly used in machine learning and natural language processing.

### Programming Language

* **Python**

### Main Libraries

* **NumPy**
* **scikit-learn**
* **matplotlib**
* **PyTorch**
* **collections / regex / os / math** from the Python standard library

### Machine Learning Techniques

* Bag-of-Words representation
* Binary feature vectorization
* Information Gain feature selection
* Probabilistic classification
* Linear classification
* Ensemble learning
* Recurrent neural networks

---

## Dataset

The project is built around the **IMDB Large Movie Review Dataset** (`aclImdb`).

The dataset follows the familiar folder-based structure:

```text
aclImdb/
├── train/
│   ├── neg/
│   └── pos/
└── test/
    ├── neg/
    └── pos/
```

Each review is stored as a separate text file. The project reads these files directly from the filesystem.

### Labels

In the current implementation, labels are encoded as:

* `1` for **negative** reviews
* `0` for **positive** reviews

### Important Note

The dataset itself is **not included** in this repository. To run the experiments, you must download the IMDB dataset separately and update the dataset paths in the scripts if necessary.

---

## General Workflow

Although the repository includes several different scripts, most experiments follow the same overall machine learning pipeline.

### 1. Data Loading

The scripts read review files from the IMDB train/test directories.

### 2. Label Assignment

Each review is assigned a sentiment label based on whether it comes from the `neg` or `pos` folder.

### 3. Vocabulary Construction

A vocabulary is created from the training data, often after frequency-based filtering.

### 4. Feature Extraction

Text is converted into feature vectors using a **Bag-of-Words** approach, usually with `CountVectorizer` and binary features.

### 5. Model Training

A classifier is trained using either a custom implementation or a library implementation.

### 6. Prediction

The model predicts class labels on train, development, and/or test sets.

### 7. Evaluation

The results are evaluated using metrics such as:

* precision
* recall
* F1-score
* classification reports

### 8. Visualization

Some scripts generate plots to compare model performance across different training sizes or experimental settings.

---

## Data Preprocessing and Vocabulary Handling

A large part of the project focuses on preparing the text data in a consistent way for multiple classifiers.

### File Sorting

The review files are sorted using the numeric prefix in their filenames, ensuring a consistent ordering of the dataset.

### Review Representation

Each review is typically stored as a structure containing:

* the class label
* the filename
* the review text content

### Vocabulary Construction

Several scripts include helper functions to build a vocabulary from the training corpus.

The typical logic includes:

* reading all training reviews
* counting word frequencies
* sorting the vocabulary
* filtering extremely common or very rare words
* selecting a fixed subset of terms

This preprocessing stage is important because it determines the features used by all classical machine learning models.

### Feature Representation

Most classical models use:

* **Bag-of-Words** features
* often **binary** word presence instead of raw counts

Some experiments also apply **Information Gain (IG)** in order to select the most informative features.

---

# Part A — Custom Classical Machine Learning Models

A major part of the repository focuses on implementing classical text classification models from scratch.

These custom implementations are used not only for prediction, but also to better understand the internal mechanics of each learning algorithm.

---

## Naive Bayes

### File: `Bayes.py`

This file contains the custom implementation of a **Naive Bayes classifier** for text classification.

It also includes supporting utility functions for:

* loading review files
* combining positive and negative examples
* building vocabularies
* preparing data for vectorization

### Core Idea

Naive Bayes assumes that features are conditionally independent given the class label.

In this project:

* class prior probabilities are estimated from the training set
* feature probabilities are estimated for each class
* **Laplace smoothing** is used to avoid zero probabilities

### Main Methods

* `train(X, Y)` computes class priors and feature likelihoods
* `provlepsi(X)` calculates log-probabilities and returns class predictions
* `predict(X)` is a wrapper for prediction

### Why It Matters

Naive Bayes is one of the most classic baseline models for text classification, and this implementation demonstrates how probabilistic reasoning can be applied to natural language tasks.

---

## Logistic Regression

### File: `Logistic_Regression.py`

This file contains a custom implementation of **Logistic Regression**.

Like `Bayes.py`, it also includes utility functions related to:

* dataset loading
* vocabulary construction
* data preparation

### Core Idea

Logistic Regression models the probability of a binary class using a linear combination of features followed by the sigmoid function.

### Main Components

* learning rate
* number of training iterations
* regularization parameter
* model weights

### Main Methods

* `sigmoid(z)` computes the logistic activation
* `fit(X, y)` learns model weights iteratively using gradient-based optimization
* `predict(X)` converts predicted probabilities to binary class labels

### Why It Matters

Logistic Regression is a powerful and interpretable linear classifier often used in text classification. This implementation highlights how optimization and linear decision boundaries can be applied to sparse text features.

---

## AdaBoost

### File: `AdaBoost.py`

This file contains the custom implementation of **AdaBoost**.

It also includes helper routines for:

* loading the dataset
* combining positive and negative examples
* vocabulary construction

### Core Idea

AdaBoost is an ensemble learning algorithm that combines many weak learners into a stronger classifier.

In this project:

* each weak learner is a shallow decision tree stump
* training begins with uniform sample weights
* weights are updated based on classification errors
* stronger emphasis is placed on difficult training examples

### Main Methods

* `fit(X, y)` trains multiple weak learners and stores their weights
* `predict(X)` produces a weighted vote across all trained learners

### Why It Matters

AdaBoost adds an ensemble perspective to the project, showing how multiple weak models can be combined to improve classification performance.

---

# Part B — Comparative Experiments with scikit-learn

Another major goal of the repository is to compare custom implementations with equivalent models from **scikit-learn**.

This allows the project to function not only as an implementation exercise, but also as an empirical study of correctness and performance.

---

## Custom Naive Bayes vs GaussianNB

### File: `main_bayes_part2.py`

This script compares:

* the custom `Bayes` classifier
* `GaussianNB` from scikit-learn

### Additional Functionality

This part also computes **Information Gain** for features.

The script:

* loads and prepares the data
* calculates feature importance using IG
* selects top features
* trains both classifiers
* evaluates them on train and development data
* plots comparative performance curves

### Why It Matters

This experiment studies not only classification performance, but also the effect of feature selection and the relationship between a custom implementation and a standard library model.

---

## Custom Logistic Regression vs sklearn LogisticRegression

### File: `main_partB_Logistic.py`

This script compares:

* the custom `Logistic_Regression`
* `LogisticRegression` from scikit-learn

### Workflow

* train/dev/test partition handling
* vocabulary generation
* vectorization
* model training for increasing data sizes
* metric collection on multiple datasets
* comparative plotting

### Why It Matters

This part helps validate the custom model and illustrates how custom optimization compares with a well-established production-grade library implementation.

---

## Custom AdaBoost vs sklearn AdaBoostClassifier

### File: `Adaboost_main_B.py`

This script compares:

* the custom `AdaBoost`
* `AdaBoostClassifier` from scikit-learn

### Workflow

* feature extraction
* repeated training with multiple dataset sizes
* train and development evaluation
* plotting of precision, recall, and F1-score

### Why It Matters

This part provides a strong comparative evaluation of an ensemble classifier and shows whether the custom design behaves similarly to the library-based version.

---

# Experiment Scripts and Their Roles

The repository includes several scripts, each dedicated to a specific experiment.

## `main_bayes.py`

Main experiment runner for the custom Naive Bayes model.

It performs:

* train/dev/test preparation
* vocabulary construction
* vectorization
* training of the custom classifier
* evaluation with standard metrics
* learning curve visualization

## `main_bayes_part2.py`

Extended Naive Bayes experiment with:

* Information Gain feature selection
* comparison against `GaussianNB`
* train/dev evaluation and plotting

## `main_partA_Logistic.py`

Main experiment runner for the custom Logistic Regression model.

It performs:

* data loading
* train/dev/test processing
* vocabulary and feature creation
* model training
* evaluation and plotting

## `main_partB_Logistic.py`

Comparative Logistic Regression experiment between:

* custom implementation
* scikit-learn implementation

## `Adaboost_main.py`

Main experiment runner for the custom AdaBoost model.

It performs:

* data loading
* Information Gain-based feature selection
* feature vectorization
* AdaBoost training
* evaluation on train and test sets
* performance plotting

## `Adaboost_main_B.py`

Comparative AdaBoost experiment between:

* custom implementation
* scikit-learn implementation

## `ParrtG.py`

Part G of the project, focused on the neural model using an RNN in PyTorch.

---

# Part G — Neural Network Approach with PyTorch

The repository also includes a neural-network-based approach to sentiment classification.

This part extends the project beyond traditional machine learning and explores sequence modeling with deep learning.

---

## IMDB Dataset Wrapper for PyTorch

### File: `IMDBDDataset.py`

This file defines the data structures needed for the PyTorch-based experiment.

### `IMDBDataset`

A subclass of `torch.utils.data.Dataset` that:

* receives raw text reviews and labels
* tokenizes each review
* converts words into integer IDs using a vocabulary dictionary
* truncates or pads sequences to a fixed maximum length
* returns tensors that can be fed into a neural network

This class is the bridge between raw text data and the neural model.

---

## Simple RNN Model

### File: `IMDBDDataset.py`

The same file also defines a neural sequence classifier called `SimpleRNN`.

### Architecture

The model contains:

* an embedding layer
* an `nn.RNN` recurrent layer
* dropout regularization
* a linear output layer

### Forward Flow

The input reviews are:

1. converted into embeddings
2. processed through the recurrent layer
3. pooled across time
4. passed through dropout
5. mapped to the final prediction layer

### Why It Matters

This part introduces a neural NLP approach and allows the project to compare classical Bag-of-Words models with a sequence-based deep learning model.

---

## Neural Experiment Script

### File: `ParrtG.py`

This script runs the RNN experiment.

It performs:

* loading train/test data
* creation of a development split
* vocabulary creation
* conversion of text into indexed sequences
* creation of `Dataset` and `DataLoader` objects
* RNN model initialization
* training for multiple epochs
* tracking of train and dev loss
* plotting of learning curves
* test-set evaluation with classification metrics

### Metrics Reported

This script reports:

* classification report
* precision, recall, and F1-scores
* micro average
* macro average
* train/dev loss curves

---

# Information Gain Feature Selection

A particularly important aspect of the project is the use of **Information Gain (IG)** in several experiments.

Information Gain is used to evaluate how informative each feature is for the class label.

### Role in the Project

IG is used to:

* rank vocabulary terms
* select the most useful features
* reduce noise in the feature space
* improve the representation before training certain classifiers

This makes the project more than a basic classification implementation. It also includes an element of **feature engineering and feature selection**, which is an important topic in machine learning.

---

# Evaluation Strategy

The project places strong emphasis on performance evaluation.

### Common Metrics Used

Across the scripts, the following evaluation metrics are commonly used:

* precision
* recall
* F1-score
* detailed classification report

### Additional Evaluation Elements

Some scripts also include:

* train vs dev comparisons
* train vs test comparisons
* comparisons across increasing training sizes
* visual plots of precision, recall, and F1-score
* train and dev loss curves for the RNN

### Why This Matters

This evaluation strategy makes the project more rigorous and more useful academically, because it emphasizes not just implementation but also systematic comparison and analysis.

---

# What This Project Demonstrates

This repository demonstrates the full workflow of a practical machine learning project for NLP.

It covers:

## 1. Raw Text Processing

Reading review files directly from a real dataset structure.

## 2. Feature Engineering

Building vocabularies, vectorizing text, and selecting informative features.

## 3. Custom Model Design

Implementing classical machine learning algorithms manually.

## 4. Comparative Evaluation

Comparing custom algorithms with standard scikit-learn implementations.

## 5. Neural Modeling

Applying a recurrent neural network for the same sentiment analysis problem.

## 6. Experimental Analysis

Measuring and visualizing precision, recall, F1-score, and loss.

Together, these stages make the project a strong example of end-to-end sentiment analysis experimentation.

---

# How to Run the Project

Because the repository contains several independent experiment scripts, each part can be run separately depending on the model you want to evaluate.

## Prerequisites

Make sure you have installed:

* Python 3.x
* NumPy
* scikit-learn
* matplotlib
* PyTorch

Example:

```bash
pip install numpy scikit-learn matplotlib torch
```

---

## Dataset Setup

Before running any script, download the IMDB dataset and make sure it is available locally.

You may need to update the hardcoded dataset paths in the scripts so they match your machine.

The expected directory pattern is:

```text
aclImdb/
├── train/
│   ├── neg/
│   └── pos/
└── test/
    ├── neg/
    └── pos/
```

---

## Example Execution

### Run Naive Bayes experiment

```bash
python main_bayes.py
```

### Run Naive Bayes comparison experiment

```bash
python main_bayes_part2.py
```

### Run Logistic Regression experiment

```bash
python main_partA_Logistic.py
```

### Run Logistic Regression comparison experiment

```bash
python main_partB_Logistic.py
```

### Run AdaBoost experiment

```bash
python Adaboost_main.py
```

### Run AdaBoost comparison experiment

```bash
python Adaboost_main_B.py
```

### Run RNN experiment

```bash
python ParrtG.py
```

---

# Why This Project Is Interesting

This project stands out because it does not rely on just one machine learning method.

Instead, it brings together:

* custom implementations of classical classifiers
* comparisons with professional ML libraries
* feature selection with Information Gain
* deep learning with an RNN
* consistent evaluation across multiple experiments

This makes it both a practical NLP project and a strong academic study in machine learning and artificial intelligence.

---

# Learning Outcomes

By studying or extending this project, one can better understand:

* how text classification works in practice
* how Bag-of-Words features are constructed
* how Naive Bayes, Logistic Regression, and AdaBoost behave on text data
* how custom implementations compare with library models
* how feature selection affects performance
* how sequence-based neural networks differ from classical ML models
* how to evaluate NLP classifiers systematically

---

# Possible Extensions

The structure of the project makes it suitable for many future improvements.

Possible extensions include:

* replacing hardcoded paths with configuration files or command-line arguments
* adding preprocessing steps such as stopword removal or stemming
* using TF-IDF instead of binary Bag-of-Words features
* adding more advanced neural architectures such as LSTM or GRU
* integrating pretrained embeddings
* performing hyperparameter tuning with validation search
* exporting results automatically to files
* creating a unified experiment runner for all models

---

# Repository Goals

This repository showcases:

* sentiment analysis on the IMDB dataset
* custom machine learning implementations in Python
* comparison with scikit-learn baselines
* feature selection using Information Gain
* neural sequence modeling with PyTorch
* experimental evaluation of NLP classifiers

---

## Authors


* Konstantinos Gougas




---

## Final Notes

This project provides a complete educational view of sentiment analysis, from classical probabilistic and linear models to ensemble techniques and neural networks. By combining implementation, experimentation, feature selection, and comparison against standard libraries, it serves as a strong academic repository for machine learning and natural language processing.
