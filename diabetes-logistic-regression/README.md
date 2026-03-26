# Multiclass Diabetes Risk Classification using Logistic Regression

## Overview

This project focuses on predicting diabetes risk levels using a multiclass classification approach. A logistic regression model was implemented alongside stratified cross-validation to evaluate performance under class imbalance conditions.

## Dataset

The dataset was obtained from Kaggle and is based on ENSANUT 2024 data published by the Instituto Nacional de Salud Pública.

* Observations: 2549
* Variables: 24
* Final dataset after preprocessing: 1815 observations

The dataset includes demographic, anthropometric, and biochemical variables such as age, BMI, glucose levels, cholesterol, and insulin.

## Problem Formulation

The original problem was reformulated from regression into a multiclass classification problem.

Target variable:

* `riesgo_diabetes_cat`

  * 0 → Low risk
  * 1 → Moderate risk
  * 2 → High risk

## Methodology

* Data cleaning and preprocessing
* Handling missing values
* Feature engineering (BMI recalculation, encoding)
* Stratified train-test split (80/20)
* Logistic Regression with class weighting
* Stratified K-Fold Cross Validation (k=5)

## Key Challenge

The dataset presents severe class imbalance:

* Class 0: ~72%
* Class 2: ~27%
* Class 1: ~0.5%

This significantly affects model performance, particularly for the minority class.

## Results

### Cross-validation performance:

* Accuracy: ~0.76
* F1 Macro: ~0.57
* F1 Weighted: ~0.83

### Test performance:

* Accuracy: 0.71

The model performs well for majority classes but struggles with the minority class due to extreme imbalance.

## Key Insights

* Logistic regression shows strong discrimination between low and high risk (AUC > 0.97).
* Performance on the moderate risk class is unstable due to low representation.
* Cross-validation provides a reliable estimate of generalization performance.

## Files

* [Notebook](./diabetes_logistic.ipynb)
* [HTML Report](./diabetes_logistic.html)
* [Dataset](./data.csv)

## Conclusion

This project demonstrates the importance of handling class imbalance in medical classification problems and highlights the limitations of standard models when dealing with extremely rare classes.

Future work may include:

* Resampling techniques (SMOTE)
* Alternative models (Random Forest, Boosting)
* Threshold tuning
