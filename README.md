# Online Shoppers Purchase Prediction

An end-to-end Machine Learning project that predicts whether an online visitor is likely to make a purchase based on their browsing behavior and session-level information.

The project covers the complete ML workflow — from data preprocessing and model selection to hyperparameter tuning and deployment using Streamlit.

##  Live Demo

**Streamlit App:** [Add your deployed Streamlit URL here]

## Problem Statement

Online businesses receive large amounts of visitor and session data, but not every visitor completes a purchase.

This project builds a classification model to predict the likelihood of a visitor generating revenue (`Revenue = True/False`).

The prediction can help businesses better understand high-intent visitors and support data-driven marketing and customer engagement strategies.

## Objective

* Predict whether an online shopping session will result in a purchase.
* Handle numerical and categorical features appropriately.
* Address class imbalance during model training.
* Optimize the model using cross-validation and hyperparameter tuning.
* Deploy the trained ML pipeline as an interactive web application.

##  Machine Learning Approach

### 1. Data Preparation

* Separated features (`X`) and target (`y`).
* Split the dataset into training and testing sets.
* Used stratified splitting to preserve the target-class distribution.

### 2. Feature Preprocessing

Different preprocessing techniques were applied based on feature type:

* **Numerical features:** StandardScaler
* **Categorical features:** OneHotEncoder
* `handle_unknown="ignore"` was used to safely handle unseen categorical values.

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature group.

### 3. Model

A **Decision Tree Classifier** was used for the classification task.

The model was configured with constraints such as:

* `max_depth`
* `min_samples_leaf`
* `class_weight="balanced"`

These settings help control overfitting and address class imbalance.

### 4. Hyperparameter Tuning

`GridSearchCV` was used to search for better Decision Tree hyperparameters.

The model was evaluated using:

* **F1 Score**
* **5-Fold Cross-Validation**

F1 Score was selected because the target classes are imbalanced and both precision and recall are important.

### 5. ML Pipeline

The complete preprocessing and model workflow was combined using Scikit-learn's `Pipeline`.

```text
Raw Input
    ↓
ColumnTransformer
    ├── Numerical → StandardScaler
    └── Categorical → OneHotEncoder
    ↓
Decision Tree Classifier
    ↓
Prediction
```

This ensures that the same preprocessing steps are consistently applied during training and prediction.

## 📊 Model Evaluation

The model was evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

> Add your final test-set scores here after training.

| Metric    | Score |
| --------- | ----- |
| Accuracy  | XX%   |
| Precision | XX%   |
| Recall    | XX%   |
| F1 Score  | XX    |

## 💾 Model Persistence

The best-performing pipeline obtained from `GridSearchCV` was saved using `joblib`.

This allows the trained preprocessing + model pipeline to be reused without retraining.

```text
online_shoppers_model.joblib
```

## Deployment

The trained model was deployed using **Streamlit**, providing an interactive interface where users can enter session information and receive a purchase prediction.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

##  Project Structure

```text
Online-Shoppers-Purchase-Prediction/
│
├── app.py
├── online_shoppers_model.joblib
├── requirements.txt
├── README.md
│
└── notebooks/
    └── model_training.ipynb
```

## ⚙️ Run Locally

### Clone the repository

```bash
git clone <your-github-repository-url>
cd Online-Shoppers-Purchase-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

## Key Learnings

Through this project, I worked with:

* Classification problem formulation
* Train-test splitting with stratification
* Numerical and categorical preprocessing
* Feature transformation using `ColumnTransformer`
* Decision Tree classification
* Handling imbalanced classes
* Hyperparameter tuning with `GridSearchCV`
* Cross-validation
* F1-score based model evaluation
* Building reusable ML pipelines
* Model serialization with Joblib
* Streamlit deployment
