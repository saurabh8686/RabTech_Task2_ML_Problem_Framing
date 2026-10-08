# ML Problem Framing & Responsible Data Card

## RabTech Academy Internship — Task 2

> **Project:** ML Problem Framing & Responsible Data Card  
> **Domain:** Customer Churn Prediction  
> **Machine Learning Type:** Binary Classification  
> **Model:** Logistic Regression  
> **Status:** Educational ML Baseline / Prototype

---

## 1. Project Overview

This project was completed as part of the **RabTech Academy internship** and focuses on two important stages of a responsible machine learning workflow:

1. **ML Problem Framing**
2. **Responsible Data Documentation**

The project uses a small customer churn dataset to demonstrate how a machine learning problem can be defined, analyzed, modeled, evaluated, and documented responsibly.

The objective is to determine whether customer-level information can be used to identify customers who may have churned or may be considered for retention-related analysis.

The project does not treat the machine learning model as an automated decision-making system. Instead, the model is presented as an **educational baseline that could support human review**.

Because the supplied dataset contains only 12 records, the experiment is intentionally treated as a learning exercise rather than a production-ready machine learning solution.

---

# 2. Project Objectives

The major objectives of this project are:

- Define a clear machine learning problem.
- Identify the unit of observation.
- Define the target variable.
- Identify appropriate model features.
- Exclude identifier fields from model training.
- Inspect the quality of the supplied dataset.
- Analyze the distribution of the target variable.
- Establish a non-ML baseline.
- Build a Logistic Regression baseline model.
- Apply appropriate preprocessing to numerical and categorical features.
- Split the data into training and testing sets.
- Evaluate the model using multiple classification metrics.
- Generate a confusion matrix.
- Compare the machine learning model with the baseline.
- Document limitations of the experiment.
- Identify responsible ML risks.
- Define safeguards and monitoring considerations.
- Prepare a Responsible Data Card.

---

# 3. Business Problem

Customer churn refers to customers who stop using or discontinue a service.

From a business perspective, identifying customers who may be at risk of churn can help an organization consider proactive retention activities.

For this educational project, the proposed problem is:

> **Can customer-level information be used to classify whether a customer belongs to the churned or non-churned class?**

The model output can be viewed as a signal that may support human review.

The model should **not** be used as the sole basis for automatically deciding what action should be taken against a customer.

---

# 4. ML Problem Definition

## 4.1 Problem Type

This project is formulated as a:

**Binary Classification Problem**

The model predicts one of two classes:

| Value | Meaning |
|---|---|
| `0` | Not Churned |
| `1` | Churned |

---

## 4.2 Unit of Observation

The unit of observation is:

> **One customer**

Each row in the dataset represents one customer.

---

## 4.3 Target Variable

The target variable is:

```text
churned
```

Interpretation:

```text
0 → Not Churned
1 → Churned
```

---

## 4.4 Input Features

The model uses the following features:

| Feature | Description | Type | Used in Model |
|---|---|---|---|
| `customer_id` | Customer identifier | Identifier | No |
| `tenure_months` | Customer tenure in months | Numerical | Yes |
| `support_tickets` | Number of support tickets | Numerical | Yes |
| `monthly_spend_inr` | Monthly customer spending in INR | Numerical | Yes |
| `last_login_days` | Number of days since last login | Numerical | Yes |
| `plan_type` | Customer subscription plan | Categorical | Yes |
| `churned` | Churn outcome | Target | No |

---

# 5. Why `customer_id` Is Excluded

The `customer_id` field is an identifier rather than a meaningful customer behavior feature.

Therefore, it is excluded from model training.

The model uses:

```python
X = df.drop(columns=["customer_id", "churned"])
```

The target is:

```python
y = df["churned"]
```

This prevents the identifier from being unnecessarily passed to the machine learning model.

---

# 6. Dataset Description

The supplied dataset contains:

```text
Rows: 12
Columns: 7
```

The dataset contains customer-level information including:

- Customer tenure
- Support tickets
- Monthly spending
- Last login activity
- Subscription plan
- Churn outcome

The dataset is intentionally small and is suitable for demonstrating an ML workflow, but it is not sufficient for establishing reliable production performance.

---

# 7. Dataset Structure

The dataset contains the following columns:

```text
customer_id
tenure_months
support_tickets
monthly_spend_inr
last_login_days
plan_type
churned
```

---

# 8. Data Quality Checks

Several basic data-quality checks were performed before model training.

## 8.1 Dataset Shape

The dataset contains:

```text
12 records
7 columns
```

---

## 8.2 Missing Values

The dataset was checked for missing values.

Result:

```text
No missing values detected
```

---

## 8.3 Duplicate Records

The dataset was checked for duplicate rows.

Result:

```text
0 duplicate rows
```

---

## 8.4 Target Distribution

The target distribution is:

| Churn Status | Count | Percentage |
|---|---:|---:|
| Not Churned (`0`) | 7 | 58.33% |
| Churned (`1`) | 5 | 41.67% |
| **Total** | **12** | **100%** |

This distribution is relatively close to balanced for this tiny dataset, although the absolute number of observations remains very small.

---

# 9. Exploratory Dataset Inspection

The notebook performs basic exploratory inspection using pandas.

The following information is examined:

- Dataset shape
- Column names
- Missing values
- Duplicate records
- Target distribution
- Numerical statistics

A class-distribution visualization is also generated.

Output:

```text
outputs/class_distribution.png
```

---

# 10. Train/Test Split

The dataset is divided into training and testing subsets.

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

The resulting split is:

| Dataset | Records |
|---|---:|
| Training set | 9 |
| Testing set | 3 |
| Total | 12 |

Stratification is used so that the class distribution is maintained as much as possible between the training and testing subsets.

---

# 11. Data Preprocessing

The project contains both numerical and categorical features.

Therefore, separate preprocessing strategies are applied.

## 11.1 Numerical Features

Numerical features:

```text
tenure_months
support_tickets
monthly_spend_inr
last_login_days
```

The numerical preprocessing pipeline uses:

1. Median imputation
2. StandardScaler

Conceptually:

```text
Numerical Data
      ↓
Median Imputation
      ↓
Standard Scaling
```

---

## 11.2 Categorical Features

Categorical feature:

```text
plan_type
```

The categorical preprocessing pipeline uses:

1. Most-frequent imputation
2. One-hot encoding

Conceptually:

```text
Categorical Data
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
```

`handle_unknown="ignore"` is used with the encoder so that previously unseen categories do not cause an encoding failure during prediction.

---

# 12. Preprocessing Pipeline

The numerical and categorical transformations are combined using a `ColumnTransformer`.

This provides a structured preprocessing workflow before the model receives the data.

The overall workflow is:

```text
                    Customer Dataset
                           |
              +------------+------------+
              |                         |
        Numerical Features        Categorical Feature
              |                         |
       Median Imputation        Most-Frequent Imputation
              |                         |
        StandardScaler           One-Hot Encoding
              |                         |
              +------------+------------+
                           |
                  Logistic Regression
                           |
                       Prediction
```

---

# 13. Baseline Model

Before evaluating the machine learning model, a simple non-ML baseline is established.

## Majority-Class Baseline

The majority class in the dataset is:

```text
0 — Not Churned
```

The baseline predicts class `0` for every customer.

Since 7 out of 12 records belong to class `0`, the baseline accuracy is:

```text
7 / 12 = 58.33%
```

Baseline results:

| Metric | Score |
|---|---:|
| Accuracy | 58.33% |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 Score | 0.00 |

The zero recall is particularly important because the baseline does not identify any churned customers.

---

# 14. Machine Learning Model

The machine learning baseline used in this project is:

> **Logistic Regression**

Logistic Regression is suitable for demonstrating a binary classification workflow.

The model is implemented inside a preprocessing and classification pipeline.

Conceptually:

```text
Raw Dataset
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Evaluation
```

---

# 15. Model Training

The model is trained using the training data:

```text
Training records = 9
```

The preprocessing transformations and classifier are contained inside a single pipeline.

This helps keep preprocessing and model training connected and reduces the risk of applying inconsistent transformations.

---

# 16. Model Evaluation

The trained model is evaluated on:

```text
Testing records = 3
```

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification report
- Confusion matrix

---

# 17. Model Results

The Logistic Regression model produced the following results on the three-record test set:

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

The predicted values matched the three test observations in this particular train/test split.

---

# 18. Important Interpretation of the Results

The reported 100% metrics must be interpreted carefully.

The dataset contains only:

```text
12 total records
```

and the test set contains only:

```text
3 records
```

Therefore, a single correct or incorrect prediction can substantially change the reported metrics.

The 100% result should **not** be interpreted as evidence that the model will achieve 100% performance on new customers.

The result demonstrates that the model successfully classified the very small test set used in this experiment.

It does not establish production-level generalization.

---

# 19. Confusion Matrix

A confusion matrix is generated to examine the classification results.

Output:

```text
outputs/confusion_matrix.png
```

For the three-record test set, the predictions correspond to:

```text
Actual:    [1, 0, 0]
Predicted: [1, 0, 0]
```

The resulting test-set confusion matrix contains:

```text
[[2, 0],
 [0, 1]]
```

This corresponds to:

- 2 correctly classified non-churned customers
- 1 correctly classified churned customer
- 0 false positives
- 0 false negatives

Again, these values are based on only three test observations.

---

# 20. Model Comparison

The project compares the non-ML majority-class baseline with Logistic Regression.

The comparison includes:

- Accuracy
- Precision
- Recall
- F1 Score

The comparison is saved as:

```text
outputs/model_comparison.csv
```

A visual comparison is saved as:

```text
outputs/model_comparison.png
```

The comparison demonstrates the difference between a simple majority-class strategy and a machine learning baseline.

---

# 21. Model Evaluation Metrics

## Accuracy

Accuracy represents the proportion of predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

It provides an overall view of classification correctness.

---

## Precision

Precision measures how many predicted positive cases were actually positive.

For this project, it can be interpreted as:

> Of the customers predicted as churned, how many were actually churned?

---

## Recall

Recall measures how many actual positive cases were successfully identified.

For this project:

> Of the customers who actually churned, how many were identified by the model?

Recall is particularly relevant when missed churn cases are considered important.

---

## F1 Score

F1 Score combines precision and recall into a single metric.

It is useful when both false positives and false negatives matter.

---

# 22. Error Analysis

Two important types of prediction errors are considered.

## False Positive

A false positive occurs when:

```text
Actual = Not Churned
Prediction = Churned
```

Potential consequence:

- Unnecessary retention outreach
- Unnecessary incentives
- Additional operational cost
- Possible customer inconvenience

---

## False Negative

A false negative occurs when:

```text
Actual = Churned
Prediction = Not Churned
```

Potential consequence:

- A potentially valuable retention opportunity may be missed
- The customer may not receive appropriate outreach

---

# 23. Proposed Decision Use

The proposed use of the model is to:

> Identify customers who may warrant consideration for retention outreach.

The model should provide a signal for human review rather than automatically determining the action taken for a customer.

A human decision-maker should be able to review relevant information before taking action.

---

# 24. Human Review

Human review is an important safeguard.

The model should not independently:

- Cancel a customer account
- Deny a service
- Change customer eligibility
- Apply penalties
- Make irreversible customer decisions

The model output should be treated as decision-support information.

---

# 25. Abstention and Data Quality

A responsible system should avoid making a prediction when input data does not meet expected quality requirements.

Potential abstention conditions include:

- Missing required input values
- Invalid feature ranges
- Failed data-quality checks
- Unexpected feature categories
- Significant distribution changes
- Insufficient model confidence

In such situations, manual review or a documented non-ML procedure should be considered.

---

# 26. Responsible Data Card

A separate Responsible Data Card is included in:

```text
reports/responsible_data_card.md
```

The data card documents:

- Dataset purpose
- Provenance and permission
- Population and representation
- Features and target
- Data quality
- Risks and safeguards
- Intended evaluation

The supplied template specifically emphasizes documenting these areas before relying on a dataset for machine learning.

---

# 27. Data Provenance

The supplied materials do not provide sufficient information about the original data collection process.

Therefore, the following information is treated as unknown:

- Original data collection methodology
- Original data owner
- Consent process
- Licensing restrictions
- Sampling methodology
- Broader population represented by the dataset

The dataset was supplied for the internship exercise, but that alone does not establish that it is representative of a real-world customer population.

---

# 28. Population and Representation

The dataset contains only 12 customer records.

The available data does not establish whether these customers represent a larger customer population.

There is also insufficient information to determine whether particular customer groups are:

- Overrepresented
- Underrepresented
- Completely absent

Therefore, generalization beyond the supplied records cannot be established.

---

# 29. Fairness Considerations

Subgroup fairness cannot be meaningfully evaluated from the supplied dataset because appropriate subgroup information is not provided.

For example, the dataset does not provide sufficient information for a reliable subgroup comparison across demographic or other protected characteristics.

Therefore, the project does not claim that the model is fair across real-world customer groups.

Additional representative data and appropriate governance would be required for meaningful fairness analysis.

---

# 30. Data Leakage Considerations

Potential leakage is considered as part of responsible ML analysis.

The dataset does not provide detailed timestamps describing exactly when each feature was collected relative to the churn outcome.

Therefore, it is not possible to fully establish a robust temporal validation strategy from the supplied materials.

For a production system, features should only use information that would have been available at the time a prediction was made.

---

# 31. Temporal Validation Limitation

The supplied dataset does not contain sufficient temporal information to perform a reliable time-based train/test evaluation.

A production churn prediction system would ideally distinguish:

```text
Past Information
       ↓
Prediction Date
       ↓
Future Outcome
```

This project cannot establish that relationship robustly because the necessary temporal information is not available.

---

# 32. Risk Register

A separate risk register is included:

```text
reports/risk_register.md
```

The risk register identifies potential issues including:

- Tiny dataset size
- Unknown provenance
- Potential data leakage
- False negatives
- False positives
- Distribution shift
- Calibration concerns
- Limited subgroup information
- Automation misuse
- Input data-quality problems
- Model degradation
- Unintended use

Each risk is documented with appropriate mitigation or monitoring considerations.

---

# 33. Monitoring Considerations

If a similar system were ever developed using a sufficiently large and representative dataset, monitoring could include:

### Data Monitoring

- Missing-value rates
- Feature distributions
- Unexpected categories
- Input ranges

### Prediction Monitoring

- Prediction distribution
- Churn prediction rate
- Confidence distribution

### Performance Monitoring

- Accuracy
- Precision
- Recall
- F1 Score
- False-positive rate
- False-negative rate

### Calibration Monitoring

Predicted probabilities should be checked to determine whether they correspond reasonably to observed outcomes.

### Fairness Monitoring

Where appropriate subgroup data is lawfully available and suitable for analysis, performance should be examined across relevant groups.

---

# 34. Rollback and Failure Handling

A production system should have a mechanism to stop automated predictions if serious data or performance problems are detected.

Potential safeguards include:

1. Disable automated predictions.
2. Investigate the data-quality problem.
3. Review model performance.
4. Use manual review or a documented non-ML procedure.
5. Retrain or update the model only after appropriate validation.

The current project is not deployed to production.

---

# 35. Project Limitations

This experiment has several major limitations.

### 1. Very Small Dataset

The dataset contains only 12 records.

### 2. Small Training Set

Only 9 records are used for training.

### 3. Small Test Set

Only 3 records are used for testing.

### 4. Unstable Evaluation

A single train/test split can produce unstable results on such a small dataset.

### 5. Unknown Representativeness

The supplied documentation does not establish that the records represent a broader customer population.

### 6. Unknown Data Collection Process

The original data collection process and consent information are not documented in the supplied materials.

### 7. No Robust Temporal Validation

Temporal information is not available for establishing a robust time-based validation strategy.

### 8. Limited Fairness Analysis

Subgroup fairness cannot be meaningfully evaluated because the required subgroup information is not provided.

---

# 36. Production Readiness

This project is **not production-ready**.

The main reasons include:

- Very small dataset
- Unknown provenance
- Unknown representativeness
- Extremely small test set
- Lack of temporal validation
- Limited fairness analysis
- No production monitoring infrastructure
- No demonstrated generalization to new customers

The model is therefore intended only as an **educational ML baseline**.

---

# 37. Technologies Used

The project was implemented using Python and common machine learning libraries.

### Programming Language

```text
Python
```

### Libraries

```text
pandas
numpy
scikit-learn
matplotlib
```

### Machine Learning Components

```text
Logistic Regression
train_test_split
ColumnTransformer
Pipeline
StandardScaler
OneHotEncoder
SimpleImputer
```

### Evaluation

```text
accuracy_score
precision_score
recall_score
f1_score
classification_report
confusion_matrix
```

---

# 38. Project Structure

```text
RabTech_Task2_ML_Problem_Framing/
│
├── data/
│   └── customer-churn-training.csv
│
├── notebooks/
│   └── baseline_model.ipynb
│
├── outputs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── model_comparison.png
│
├── reports/
│   ├── problem_framing.md
│   ├── responsible_data_card.md
│   └── risk_register.md
│
├── src/
│   └── baseline.py
│
└── README.md
```

---

# 39. File Descriptions

## `data/`

Contains the supplied customer churn dataset.

```text
customer-churn-training.csv
```

---

## `notebooks/`

Contains the main experimental notebook.

```text
baseline_model.ipynb
```

The notebook demonstrates:

- Data loading
- Data inspection
- Data preprocessing
- Train/test split
- Model training
- Model evaluation
- Baseline comparison
- Visualizations
- Limitations

---

## `src/`

Contains the Python source implementation.

```text
baseline.py
```

This script performs dataset inspection and calculates the majority-class baseline.

---

## `outputs/`

Contains generated experiment results and visualizations.

### `class_distribution.png`

Visualizes the distribution of the churn target.

### `confusion_matrix.png`

Visualizes the classification results of the Logistic Regression model.

### `model_comparison.csv`

Contains the comparison between the majority-class baseline and Logistic Regression.

### `model_comparison.png`

Visualizes the comparison of evaluation metrics.

---

## `reports/`

Contains project documentation.

### `problem_framing.md`

Documents:

- Business problem
- ML formulation
- Target
- Features
- Baseline
- Evaluation
- Decision use
- Limitations
- Conclusion

### `responsible_data_card.md`

Documents responsible data considerations.

### `risk_register.md`

Documents identified ML risks and proposed mitigations.

---

# 40. How to Run the Project

## Step 1 — Open the Project

Open the project folder in Visual Studio Code:

```text
RabTech_Task2_ML_Problem_Framing
```

---

## Step 2 — Activate the Virtual Environment

In PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Step 3 — Install Dependencies

If required:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook ipykernel
```

---

## Step 4 — Run the Baseline Script

From the project root:

```powershell
python src/baseline.py
```

This performs the dataset inspection and majority-class baseline analysis.

---

## Step 5 — Open the Notebook

Start Jupyter:

```powershell
jupyter notebook
```

Then open:

```text
notebooks/baseline_model.ipynb
```

Run the notebook cells from top to bottom.

---

# 41. Expected Baseline Output

The majority-class baseline identifies:

```text
Majority Class: 0
```

Expected accuracy:

```text
0.5833
```

or:

```text
58.33%
```

The baseline has zero recall for the churned class because it predicts every customer as not churned.

---

# 42. Expected Model Output

The Logistic Regression experiment produces:

```text
Accuracy  = 1.00
Precision = 1.00
Recall    = 1.00
F1 Score  = 1.00
```

These values correspond to the specific three-record test set used in this experiment.

They should not be interpreted as production performance.

---

# 43. Responsible Use Statement

This project is intended for:

- Educational purposes
- Demonstrating ML problem framing
- Demonstrating data preprocessing
- Demonstrating baseline modeling
- Demonstrating model evaluation
- Demonstrating responsible ML documentation

This project should **not** be used as a production customer decision system.

The model should not independently determine:

- Customer penalties
- Service termination
- Customer eligibility
- Financial decisions
- High-impact decisions
- Irreversible customer actions

---

# 44. Future Improvements

If this project were developed further, the following improvements would be necessary.

## Dataset Improvements

- Collect substantially more records.
- Establish reliable data provenance.
- Document data collection procedures.
- Verify permissions and consent.
- Ensure representative sampling.
- Include appropriate timestamps.

## Model Improvements

Potential models could be evaluated after sufficient data is available:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Other appropriate classification algorithms

Models should be compared using suitable validation procedures rather than relying on one very small test set.

## Evaluation Improvements

Future evaluation should consider:

- Cross-validation
- Time-based validation where appropriate
- Calibration
- Precision
- Recall
- F1 Score
- Confusion matrix
- Error analysis
- Appropriate subgroup evaluation

## Production Improvements

A real system would require:

- Data validation
- Model monitoring
- Drift detection
- Performance monitoring
- Access controls
- Audit logging
- Human review
- Rollback procedures
- Governance
- Documentation

---

# 45. Key Learning Outcomes

This project demonstrates the following ML concepts:

### Machine Learning

- Binary classification
- Feature selection
- Train/test splitting
- Logistic Regression
- Model evaluation
- Baseline comparison

### Data Processing

- Missing-value checking
- Duplicate checking
- Numerical preprocessing
- Categorical encoding
- Feature scaling
- Pipeline construction

### Responsible AI

- Data provenance
- Representation
- Leakage
- Fairness limitations
- False positives
- False negatives
- Human review
- Model monitoring
- Responsible deployment

### ML Documentation

- Problem framing
- Responsible Data Card
- Risk Register
- Experimental reporting
- Limitations documentation

---

# 46. Final Conclusion

This project demonstrates an end-to-end workflow for framing a customer churn machine learning problem and documenting the associated responsible ML considerations.

The workflow begins with dataset inspection and problem definition, followed by feature and target selection, data preprocessing, baseline construction, Logistic Regression training, and model evaluation.

The Logistic Regression model achieved 100% accuracy, precision, recall, and F1-score on the three-record test set. However, because the entire dataset contains only 12 records, these results have very high uncertainty and cannot be considered evidence of production-level model performance.

The most important outcome of this project is therefore not the reported model score, but the demonstration of a structured and responsible ML workflow.

The project should be treated as an **educational prototype and baseline experiment**, not as a production customer churn prediction system.

---

## Project Status

```text
✓ Problem Framing Completed
✓ Dataset Inspection Completed
✓ Data Quality Checks Completed
✓ Majority-Class Baseline Completed
✓ Feature Preprocessing Completed
✓ Logistic Regression Completed
✓ Model Evaluation Completed
✓ Confusion Matrix Generated
✓ Model Comparison Completed
✓ Responsible Data Card Completed
✓ Risk Register Completed
✓ Limitations Documented
✓ Project README Completed
```

---

## Author

**Atharva Joshi**

**RabTech Academy Internship**

**Project:** ML Problem Framing & Responsible Data Card