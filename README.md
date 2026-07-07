# Wine Quality Regression

## Overview
This project develops and compares regression models for predicting wine quality using physicochemical features. The analysis focuses on model specification, predictive accuracy, and regularization, with the goal of identifying a model that balances fit quality and generalization performance.

## Objective
The main goals of this project are to:
- predict wine quality from numerical wine characteristics,
- compare different regression model specifications,
- evaluate whether interaction terms improve predictive performance,
- apply Lasso regularization to control model complexity and reduce overfitting.

## Dataset
The dataset consists of wine samples with multiple physicochemical features and a numerical quality score as the response variable.

Examples of predictor variables include:
- acidity-related measures,
- sugar content,
- chlorides,
- sulfur dioxide measures,
- density,
- pH,
- sulphates,
- alcohol.

The data is split into training and test sets:
- `dataset/wine_training.csv`
- `dataset/wine_test.csv`

The training set is used for model fitting, while the test set is used for out-of-sample evaluation.

## Methods
This project compares several regression approaches, including:
- **Ordinary Least Squares (OLS)**
- **Quadratic regression**
- **Interaction-term regression**
- **Lasso regression with cross-validation**

The workflow includes:
- training models on the training set,
- evaluating predictive performance on the test set,
- comparing mean squared error (MSE) across model specifications,
- using regularization to reduce overfitting and improve robustness.

## Key Results
- Baseline linear regression provides a useful starting point for prediction.
- Adding nonlinear structure and interaction terms improves model flexibility.
- Interaction-based models achieve stronger predictive performance than simpler specifications.
- Lasso regularization helps control complexity and improves out-of-sample robustness.
- The best-performing model achieves the lowest test error while balancing interpretability and generalization.

## My Contribution
I implemented the regression pipeline, compared multiple model specifications, evaluated out-of-sample prediction performance, and interpreted how interaction terms and regularization affected model accuracy and model complexity.

## Repository Structure
- `HW3_helper_notebook.ipynb` — main notebook containing the analysis
- `hw3_solution.py` — supporting Python implementation
- `AMATH_482__project_3_Report.pdf` — final project report
- `dataset/wine_training.csv` — training dataset
- `dataset/wine_test.csv` — test dataset
- `figures/fig_interaction_lasso_true_vs_pred.png` — predicted vs. true values visualization
- `figures/fig_lassocv_selection_curve.png` — Lasso cross-validation selection curve

## Skills Demonstrated
- Python
- Regression Modeling
- Predictive Analytics
- Model Comparison
- Feature Interactions
- Regularization
- Cross-Validation
- Data Visualization

## Conclusion
This project shows how progressively richer regression models can improve predictive performance in a supervised learning setting. By comparing standard linear regression, interaction-based models, and Lasso regularization, the analysis highlights the tradeoff between model flexibility, interpretability, and out-of-sample performance.
