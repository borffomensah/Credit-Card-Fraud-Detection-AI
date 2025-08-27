## Fraud Detection Project Documentation

This document outlines the steps taken in this fraud detection project, including data loading, exploration, preprocessing, model training, evaluation, and deployment.

### 1. Project Motivation

The goal of this project was to build a model that could accurately detect fraudulent credit card transactions. Fraudulent transactions are a significant problem for financial institutions and individuals, leading to financial losses and damage to trust. Using machine learning, we aimed to identify patterns indicative of fraud to minimize these risks.

### 2. Data Loading and Initial Exploration

The project began by loading the credit card transaction data from a CSV file. Key libraries like `pandas` for data manipulation and `matplotlib` and `seaborn` for visualization were imported.

- The dataset was loaded using `pd.read_csv()`.
- `df.info()` was used to get a summary of the DataFrame, including data types and non-null counts. This showed that there were no missing values in this dataset.
- `df.describe()` provided a statistical summary of the numerical columns, giving insights into the distribution of the data.

Initial visualizations were created to understand the data distribution:

- A histogram of the 'Amount' column revealed that most transactions were small, with a long tail of larger amounts.
- A box plot of 'Amount' by 'Class' showed that fraudulent transactions tended to have a different distribution of amounts compared to non-fraudulent ones, although there was significant overlap.
- Histograms of the 'Time' column for all transactions and fraudulent transactions separately showed the distribution of transactions over time, highlighting potential patterns in when fraudulent activities occurred.
- A bar plot of the 'Class' distribution clearly showed the severe class imbalance, with a much larger number of non-fraudulent transactions than fraudulent ones.

### 3. Data Balancing (Oversampling)

The severe class imbalance observed in the data was a significant challenge for training a robust model. Training on imbalanced data could lead to a model that was biased towards the majority class (non-fraudulent) and performed poorly in identifying the minority class (fraudulent).

To address this, Random Oversampling was employed using the `imblearn` library.

- `RandomOverSampler` was instantiated and applied to the data.
- This technique duplicated random instances of the minority class to balance the class distribution.
- The output of `np.bincount(y_resampled)` and the bar plot of the resampled data confirmed that the classes were now balanced.

### 4. Model Preparation

Before training models, the data was prepared:

- The balanced data (`X_resampled`, `y_resampled`) was split into training and testing sets using `train_test_split`. A `stratify` split was used to maintain the balanced class distribution in both training and testing sets.
- `StandardScaler` was used to standardize the features. This was important for many machine learning algorithms that are sensitive to the scale of the input features, such as Logistic Regression. The scaler was fitted only on the training data to prevent data leakage.
- The fitted `StandardScaler` was saved using `joblib` for later use in scaling new data for predictions.

### 5. Model Training and Evaluation

Several models were trained and evaluated:

#### a) Logistic Regression (Base Model)

- A `LogisticRegression` model was instantiated and trained on the scaled training data.
- The model was evaluated using `accuracy_score`, `classification_report`, and `confusion_matrix`.
- The confusion matrix and classification report provided insights into the model's performance, showing the number of true positives, true negatives, false positives, and false negatives, as well as precision, recall, and F1-score.

#### b) Random Forest Classifier

- A `RandomForestClassifier` was instantiated and trained on the scaled training data. Random Forests are an ensemble method that can handle complex relationships in the data and are generally less sensitive to feature scaling than Logistic Regression.
- The model was evaluated using the same metrics as Logistic Regression.
- The Random Forest model demonstrated significantly higher performance metrics compared to Logistic Regression, indicating its ability to capture more complex patterns related to fraud.
- The trained Random Forest model was saved using `joblib` for future use.

#### c) LGBM Classifier

- An `LGBMClassifier` was instantiated and trained on the scaled training data. LightGBM is a gradient boosting framework that is known for its speed and efficiency, often providing high accuracy.
- The model was evaluated using the classification report and confusion matrix.
- The LGBM Classifier also showed high performance, comparable to the Random Forest model, further suggesting that more advanced models are better suited for this task.

### 6. Results and Conclusion

Based on the evaluation metrics, both the Random Forest Classifier and the LGBM Classifier performed exceptionally well on this dataset after oversampling. The high precision, recall, and F1-scores for the fraudulent class indicated that these models were effective in identifying fraudulent transactions. The confusion matrices showed a very low number of false negatives (fraudulent transactions classified as non-fraudulent), which was crucial in a fraud detection system.

While Logistic Regression provided a decent baseline, the advanced tree-based models (Random Forest and LightGBM) were able to leverage the features more effectively to distinguish between fraudulent and non-fraudulent transactions.

The choice between Random Forest and LGBM might depend on further considerations such as training time, interpretability, and fine-tuning of hyperparameters. However, both showed promising results for this fraud detection task.

This project demonstrated a typical workflow for tackling imbalanced classification problems, highlighting the importance of data balancing techniques and the use of appropriate evaluation metrics beyond just accuracy.

### 7. Model Deployment

The trained Random Forest model was deployed as a web application using Streamlit. This allows for easy interaction with the model and demonstration of its fraud detection capabilities.

You can access the deployed application here: [https://credit-card-fraud-detection-ai-2025.streamlit.app/](https://credit-card-fraud-detection-ai-2025.streamlit.app/)
