# Responsible Data Card

## RabTech Academy - AI & Machine Learning Internship

## 1. Dataset Purpose

This dataset is intended for an educational baseline model for
customer churn prediction.

The dataset may support experimentation with customer churn
classification and responsible ML problem framing.

The model must not be used as the sole basis for denying services,
changing contractual terms, or making other high-impact decisions
about customers.

---

## 2. Provenance and Permission

The dataset was supplied as part of the RabTech Academy AI & ML
internship task.

The supplied materials do not document the original data collection
process, individual consent records, licensing restrictions, or the
original data owner.

Therefore, these details are treated as unknown rather than
assumed.

The dataset should be used only for its intended educational
purpose unless additional permission is documented.

---

## 3. Population and Representation

The dataset contains 12 customer records.

The supplied dataset does not document:

- Sampling methodology
- Geographic coverage
- Demographic coverage
- Customer population
- Excluded groups
- Sampling period

Therefore, the dataset should not be assumed to represent a larger
customer population.

### Target distribution

- Non-churned customers: 7
- Churned customers: 5

Percentage:

- Non-churned: 58.33%
- Churned: 41.67%

Because the dataset is very small, these percentages should not be
treated as representative of a real customer population.

---

## 4. Features and Target

### customer_id

Customer identifier.

This field is excluded from model training because it is an
identifier and does not represent a meaningful customer behavior.

### tenure_months

Number of months associated with the customer's account.

### support_tickets

Number of support tickets associated with the customer.

### monthly_spend_inr

Customer's monthly spending amount in INR.

### last_login_days

Number of days since the customer's last login.

### plan_type

Customer subscription plan.

This is a categorical feature.

### churned

Prediction target.

Values:

- `0` = Not Churned
- `1` = Churned

---

## 5. Data Leakage Risks

Only information available at the prediction time should be used
for model training and prediction.

Information generated after the prediction point or information that
directly results from the customer's eventual churn could introduce
target leakage.

The supplied dataset does not provide detailed timestamps for all
features.

Therefore, temporal leakage cannot be completely verified.

---

## 6. Quality Checks

### Dataset Size

12 rows and 7 columns.

### Missing Values

No missing values were observed in the supplied dataset.

### Duplicate Records

No duplicate rows were observed.

### Target Distribution

- Non-churned: 7
- Churned: 5

### Train/Test Separation

Training and testing data must be separated before model evaluation.

Preprocessing operations should be fitted only on the training data
to avoid information leakage.

---

## 7. Risks and Safeguards

### Risk 1 - Very Small Dataset

The dataset contains only 12 records.

#### Impact

Model performance estimates may be unstable and have high
uncertainty.

#### Safeguard

Use the model only as an educational baseline and collect a larger
representative dataset before production deployment.

---

### Risk 2 - Unknown Data Provenance

The original collection process and consent information are not
documented in the supplied materials.

#### Impact

The legal, ethical and permission status of the data cannot be
fully established.

#### Safeguard

Document the original data source, permission and intended use
before real-world deployment.

---

### Risk 3 - False Negatives

The model may predict that a customer will not churn even though
the customer actually churns.

#### Impact

A potential retention opportunity may be missed.

#### Safeguard

Monitor recall for the churned class and review false-negative
cases.

---

### Risk 4 - False Positives

The model may predict churn for a customer who does not actually
churn.

#### Impact

The business may spend resources unnecessarily on retention
activities.

#### Safeguard

Monitor precision and define an acceptable intervention budget.

---

### Risk 5 - Data Distribution Shift

Future customers may have behavior different from the training
data.

#### Impact

Model performance may degrade over time.

#### Safeguard

Monitor feature distributions and model prediction distributions.

---

### Risk 6 - Model Misuse

Predictions may be treated as certain outcomes.

#### Impact

Users may make inappropriate decisions based on an uncertain
prediction.

#### Safeguard

Present predictions as risk signals and maintain human review for
consequential actions.

---

## 8. Intended Evaluation

### Non-ML Baseline

The majority-class baseline predicts every customer as not
churned.

Baseline accuracy:

58.33%.

### Model Metrics

The ML model will be evaluated using:

- Recall
- Precision
- F1-score
- Accuracy
- Confusion Matrix

### Calibration

Predicted probabilities should be checked for calibration before
they are used as decision-support probabilities.

### Fairness

The supplied dataset does not contain sufficient protected-group
information for a meaningful subgroup fairness evaluation.

Therefore, fairness performance cannot currently be established.

### Error Analysis

False-positive and false-negative predictions should be reviewed
to identify common error patterns.

---

## 9. Intended Use

The intended use of this dataset is educational experimentation
and responsible ML problem framing.

It can be used to demonstrate:

- Binary classification
- Data preprocessing
- Baseline modelling
- Model evaluation
- Responsible AI considerations

---

## 10. Prohibited or Unsupported Use

The model should not be used as the sole basis for:

- Denying services
- Changing contractual conditions
- Making high-impact customer decisions
- Automatically penalizing customers
- Making claims about an individual customer's future behavior

---

## 11. Overall Limitation

The dataset contains only 12 observations and lacks detailed
documentation about its original population, collection process,
sampling method, consent and timestamps.

Therefore, the model is an educational baseline and should not be
considered production-ready.

## Experimental Results

The dataset was evaluated using a majority-class baseline and a Logistic Regression model.

### Majority-Class Baseline

The majority class in the dataset is `0` (Not Churned).

- Accuracy: 58.33%
- Precision for churned class: 0.00
- Recall for churned class: 0.00
- F1-score for churned class: 0.00

### Logistic Regression

The Logistic Regression model was evaluated on a three-record test set.

- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-score: 100%

Because the test set contains only three records, these results have very high uncertainty and must not be interpreted as production-level performance.

The results are included to demonstrate the ML workflow rather than to establish a reliable model for real-world customer decisions.