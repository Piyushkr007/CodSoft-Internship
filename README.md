# MOVIE-GENRE-CLASSIFICATION
Description:

This repository implements a machine learning model to classify movie genres based on their plot summaries. It leverages TF-IDF (Term Frequency-Inverse Document Frequency) for text pre-processing and feature extraction, coupled with powerful classifiers like Naive Bayes and Logistic Regression.

Key Features:

Data Preprocessing:
Text cleaning techniques: Lowercasing, removing special characters and digits, tokenization.
Stop word removal and stemming using NLTK for improved feature representation.
TF-IDF Vectorization:
Extracts informative features by considering both word frequency and document frequency.
Configurable max_features parameter to control feature dimensionality (default: 5000).
Machine Learning Models:
Naive Bayes: A probabilistic classifier known for its simplicity and efficiency.
Logistic Regression: A versatile linear classifier for multi-class classification.
Option to explore other models like Support Vector Machines (SVMs) for potential performance enhancements.
Evaluation:
Train-test split with a customizable random state for reproducibility.
Accuracy score and classification report for comprehensive model assessment.
Usage:

Prerequisites:
Python 3.x
Required libraries: pandas, matplotlib, seaborn, nltk, scikit-learn
Install missing libraries using pip install <library_name>.
Data Preparation:
Ensure your training and test data are in CSV format with columns: 'Title', 'Genre', 'Description'.
Modify file paths (train_path and test_path) accordingly in the script.
Run the Script:
Execute python movie_genre_classification.py (replace with your actual script name).
The script will:
Load and explore the data.
Visualize genre distributions.
Perform text pre-processing and TF-IDF vectorization.
Train and evaluate the chosen machine learning models.
Print validation accuracy and classification reports.
Future Enhancements:

Explore alternative text cleaning and pre-processing techniques.
Experiment with different machine learning algorithms (SVMs, Random Forests, etc.).
Implement hyperparameter tuning for model optimization.
Integrate the model into a movie recommendation system.
License:

Specify the open-source license used for your project (e.g., MIT License, Apache License 2.0).

Contribution:

Welcome contributions and feedback! Feel free to create pull requests or raise issues on this repository.
