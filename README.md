Wine Quality Prediction

This project predicts the quality of wine based on its chemical properties. A Random Forest Classifier is used to classify wine as either "Good Quality" (quality score ≥ 7) or "Bad Quality" (quality score < 7).

Dataset

1. Source: Red wine dataset (winequality-red.csv)
2. Features: Includes various chemical properties like fixed acidity, citric acid, alcohol, etc.
3. Target: The quality column is transformed into a binary classification:
   --> Good Quality: 1 (quality ≥ 7)
   --> Bad Quality: 0 (quality < 7)

Steps

1. Data Preprocessing:
   Missing values are checked and no null values are found.
   A heatmap of feature correlations is visualized using Seaborn.
2. Model:
   The dataset is split into training and test sets using train_test_split().
   A Random Forest Classifier is trained on the training data.
3. Evaluation:
   Accuracy is calculated on both the training and test sets using accuracy_score.
4. Prediction:
   An example wine data input is used to predict whether the wine is of good or bad quality.
5. Results
   The model achieves high accuracy on both the training and test datasets.

Installation

1. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn
2. Run the script to train the model and make predictions.
