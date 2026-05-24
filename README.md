# Market-Aware Machine Learning Model for Horse Racing Outcome Prediction

## Overview

This project develops a market-aware machine learning model to predict horse racing outcomes using historical race data, horse attributes, race conditions, and betting market indicators.

The primary objective of the project is to identify likely winning horses by combining domain-specific feature engineering with advanced machine learning techniques.

The project implements a complete end-to-end machine learning pipeline including:

* Data cleaning and preprocessing
* Feature engineering
* Handling class imbalance
* Model training and hyperparameter tuning
* Probability-based race-level prediction
* Model evaluation and visualization

The final model was trained using XGBoost and evaluated using both classification metrics and race-level prediction metrics.

---

# Dataset Information

The project uses two datasets:

| Dataset     | Description                             |
| ----------- | --------------------------------------- |
| `runs.csv`  | Horse-level race participation records  |
| `races.csv` | Race-level metadata and race conditions |

### Dataset Size

* Total race entries: **79,447**
* Total races: **6,349**
* Average horses per race: **12.68**

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Google Colab

---

# Machine Learning Pipeline

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
SMOTE Class Balancing
        ↓
XGBoost Model Training
        ↓
Hyperparameter Tuning
        ↓
Probability Prediction
        ↓
Race-Level Winner Selection
        ↓
Evaluation & Visualization
```

---

# Data Preprocessing

The following preprocessing steps were performed:

* Removed columns with excessive missing values
* Merged race-level and horse-level datasets
* Filled missing categorical values using mode
* Filled missing numerical values using median
* Converted date columns to datetime format
* Renamed sectional timing columns for clarity
* Extracted numeric values from horse ratings
* Applied one-hot encoding to categorical features
* Standardized numerical features using StandardScaler

---

# Feature Engineering

The following custom features were engineered to improve predictive performance:

| Feature                    | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `average_speed`            | Horse speed calculated using distance and finish time |
| `weight_diff`              | Difference between declared and actual weight         |
| `odds_ratio`               | Ratio of win odds to place odds                       |
| `race_day_of_week`         | Day of the week extracted from race date              |
| `num_horses_in_race`       | Total horses participating in a race                  |
| `race_avg_horse_rating`    | Average horse rating within each race                 |
| `horse_rating_vs_race_avg` | Relative rating compared to race average              |

---

# Handling Class Imbalance

Horse racing datasets are highly imbalanced because only one horse wins each race.

To address this issue:

* SMOTE (Synthetic Minority Oversampling Technique) was applied
* Original class distribution:

```text
Non-winners: 58,433
Winners: 5,124
```

* Balanced class distribution after SMOTE:

```text
Non-winners: 58,433
Winners: 58,433
```

---

# Model Training

The project uses the XGBoost classifier for horse race outcome prediction.

## Hyperparameter Tuning

GridSearchCV was used to optimize model performance.

### Best Hyperparameters

```python
{
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8
}
```

---

# Final Model Performance

## Classification Performance

| Metric        | Score  |
| ------------- | ------ |
| Accuracy      | 0.9484 |
| Precision     | 0.7369 |
| Recall        | 0.5235 |
| F1 Score      | 0.6121 |
| ROC-AUC Score | 0.9695 |

### ROC-AUC Analysis

The model achieved a high ROC-AUC score of **0.9695**, demonstrating excellent discriminative ability between winning and non-winning horses across multiple classification thresholds.

---

# Optimal Threshold Analysis

The optimal classification threshold based on the F1-score was identified in the range of **0.20 to 0.25**.

At a threshold of **0.25**, the model achieved:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9331 |
| Precision | 0.5471 |
| Recall    | 0.7899 |
| F1 Score  | 0.6473 |

This threshold provides a better balance between precision and recall for identifying likely winners.

---

# Race-Level Prediction Performance

Instead of only performing binary classification, the project predicts winning probabilities for all horses in a race and selects the horse with the highest probability as the predicted winner.

## Race-Level Accuracy

* Correctly predicted the winner in approximately **19.44%** of races.

## Top-3 Accuracy

* The actual winner appeared among the model's top 3 predicted horses in approximately **20.71%** of races.

Considering the average of **12.68 horses per race**, random guessing would achieve approximately:

```text
1 / 12.68 ≈ 7.9%
```

Therefore, the proposed model performs approximately **2.5× better than random selection**.

---

# Visualizations

The project includes several visualizations for model interpretation and evaluation:

* ROC Curve
* Confusion Matrix
* Feature Importance Plot
* Predicted Probability Distribution
* Predicted Rank Distribution
* Betting Odds vs Predicted Probability

---

# Key Insights

* Betting market indicators significantly influence race outcome prediction.
* Engineered features such as average speed and horse rating differentials improved predictive performance.
* The model demonstrates strong discriminative power despite the highly competitive and uncertain nature of horse racing.
* Race-level prediction remains challenging due to multiple unpredictable real-world factors.

---

# Future Improvements

Potential future enhancements include:

* Incorporating historical horse and jockey performance statistics
* Using race-wise ranking models
* Implementing LightGBM and ensemble methods
* Applying SHAP explainability techniques
* Building betting profit simulation strategies
* Exploring deep learning approaches for sequential race analysis

---

# Conclusion

This project successfully developed a market-aware machine learning model for horse racing outcome prediction using XGBoost.

The model demonstrated strong classification capability with a ROC-AUC score of **0.9695** and achieved race-level prediction accuracy significantly higher than random guessing.

The results indicate that combining horse performance metrics, race conditions, and betting market indicators can provide valuable data-driven insights into horse race outcome prediction.

---

# Author

Developed as a machine learning and data science project focused on predictive analytics and sports outcome prediction.

Logavijay
