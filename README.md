# Mobile Subscriber Plan Recommendation (Intro to Machine Learning)

## Project Overview
Mobile carrier **Megaline** seeks to transition subscribers from legacy mobile plans to newer offerings (**Smart** vs. **Ultra**). This project builds supervised binary classification models to analyze subscriber monthly behavior metrics and automatically recommend the optimal plan.

The target objective is to build a classification model achieving a **test dataset accuracy of at least 0.75**, accompanied by model sanity checks against baseline heuristics.

---

## Technical Highlights & Key Methodologies

* **Data Partitioning & Strategy:**
  * Partitioned preprocessed subscriber behavior data (`users_behavior.csv`) into **60% Training / 20% Validation / 20% Test** sets using stratified sampling to maintain class proportions.
* **Model Exploration & Hyperparameter Tuning:**
  * **Decision Tree Classifier:** Evaluated depth ranges (`max_depth` from 1 to 10) to determine optimal tree complexity and mitigate overfitting.
  * **Random Forest Classifier:** Applied grid search over tree counts (`n_estimators` from 10 to 50) and maximum depth settings to find peak accuracy.
  * **Logistic Regression:** Trained baseline linear classification model (`solver='lbfgs'`) for comparative analysis.
* **Final Evaluation & Testing:**
  * Evaluated top-performing candidate hyperparameter configurations against the isolated test dataset to verify generalization capability.
* **Sanity Check:**
  * Verified model utility by comparing accuracy against a dummy classifier baseline (most frequent class predictor).

---

## Dataset Overview

Dataset source: `/datasets/users_behavior.csv`

| Feature | Type | Description |
| :--- | :--- | :--- |
| `calls` | Quantitative | Number of calls made during the month |
| `minutes` | Quantitative | Total call duration in minutes |
| `messages` | Quantitative | Total text messages sent |
| `mb_used` | Quantitative | Volume of internet data used in MB |
| **`is_ultra`** *(Target)* | Binary | Plan subscription status (**1 = Ultra**, **0 = Smart**) |

---

## Key Model Results & Metrics

| Model Type | Best Hyperparameters | Validation Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **Decision Tree** | `max_depth = 3` | ~0.785 | ~0.778 |
| **Logistic Regression** | `solver = 'lbfgs'` | ~0.726 | ~0.741 |
| **Random Forest** | `n_estimators = 40`, `max_depth = 8` | **~0.808** | **~0.796** |
| **Dummy Baseline** | *Most Frequent Class (Smart)* | — | ~0.693 |

> **Final Outcome:** The tuned **Random Forest Classifier** achieved the highest accuracy of **~0.796** on the test dataset, exceeding Megaline's **0.75 accuracy requirement** and outperforming the baseline sanity check (~0.693).

---

## Project Structure

```text
├── main.ipynb            # Jupyter notebook containing data splitting, hyperparameter search, and test evaluations
├── README.md             # Project documentation and summary
└── /datasets/
    └── users_behavior.csv # Megaline subscriber behavior dataset
