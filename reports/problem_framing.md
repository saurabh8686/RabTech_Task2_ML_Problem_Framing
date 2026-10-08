# ML Problem Framing Memo

## RabTech Academy - AI & Machine Learning Internship

### Project Title
Customer Churn Prediction - Responsible ML Baseline

---

## 1. Business Problem

The objective is to identify customers who may be at risk of
churning so that a business can consider proactive retention
outreach.

The machine learning system is intended to provide a risk signal
that can support decision-making rather than completely replace
human judgment.

---

## 2. Proposed Decision

The proposed business decision is whether a customer should be
flagged for potential retention outreach.

A model prediction should not automatically trigger a customer
action without appropriate human review and business rules.

---

## 3. Prediction Target

The prediction target is:

`churned`

Target values:

- `0` = Customer did not churn
- `1` = Customer churned

This is a binary classification problem.

---

## 4. Unit of Observation

One row in the dataset represents one customer.

Therefore, the unit of observation is:

**Customer**

---

## 5. Candidate Features

The candidate predictive features are:

- `tenure_months`
- `support_tickets`
- `monthly_spend_inr`
- `last_login_days`
- `plan_type`

The `customer_id` column will not be used as a predictive feature
because it is an identifier rather than a meaningful behavioral
feature.

---

## 6. Action Window

For this educational problem framing, the proposed assumption is
that the prediction is generated using information available at the
prediction date and is used to prioritize potential retention
outreach during the following 30-day intervention window.

This 30-day action window is a project assumption and is not
documented as part of the supplied dataset.

---

## 7. Non-ML Baseline

Before using machine learning, a simple majority-class baseline
will be used.

The supplied dataset contains:

- 7 non-churned customers
- 5 churned customers

Therefore, the majority class is:

**0 = Not Churned**

The non-ML baseline predicts every customer as not churned.

Expected baseline accuracy:

7 / 12 = 58.33%

This provides a simple reference point against which the ML model
can be compared.

---

## 8. Proposed ML Baseline

A Logistic Regression classifier will be used as the initial
machine learning baseline.

The preprocessing pipeline will:

1. Separate features and target.
2. Remove the customer identifier.
3. Handle numerical features.
4. Scale numerical features.
5. One-hot encode the categorical `plan_type` feature.
6. Train Logistic Regression.

---

## 9. Evaluation Metrics

The primary model metric will be:

### Recall for Churned Customers

Recall measures how many of the customers who actually churned
were correctly identified by the model.

Secondary metrics:

- Precision
- F1-score
- Accuracy
- Confusion Matrix

---

## 10. False Positive Cost

A false positive occurs when the model predicts that a customer
will churn but the customer does not actually churn.

Potential consequences include:

- Unnecessary retention outreach
- Unnecessary discounts or incentives
- Additional operational cost

Therefore, precision should also be monitored.

---

## 11. False Negative Cost

A false negative occurs when the model predicts that a customer
will not churn but the customer actually churns.

Potential consequence:

The business may miss an opportunity to proactively engage and
retain the customer.

Therefore, recall for the churned class is particularly important.

---

## 12. Human Review

Model predictions should be treated as risk signals rather than
certain predictions.

A human review step should be used before high-impact retention
actions are taken.

---

## 13. Abstention

The system should be able to abstain from making an automated
recommendation when:

- The prediction confidence is insufficient.
- Required input data is missing.
- The input is outside the validated data range.
- Data quality checks fail.

In such cases, the case should be sent for manual review.

---

## 14. Monitoring

Potential monitoring checks include:

- Missing-value rate
- Feature distribution
- Prediction distribution
- Recall
- Precision
- False-positive rate
- False-negative rate
- Model calibration

Monitoring should be performed regularly if the system is ever
used beyond the educational setting.

---

## 15. Rollback Conditions

Automated predictions should be disabled if:

- Required data quality checks fail.
- Model performance falls below an agreed validation threshold.
- Significant data distribution changes are detected.
- The prediction pipeline produces unexpected outputs.

The fallback should be manual review or the documented non-ML
baseline.

---

## 16. Limitations

The supplied dataset contains only 12 customer records.

Therefore, model performance estimates from this dataset have high
uncertainty and should not be treated as evidence of production
performance.

The supplied dataset also does not document:

- Original sampling methodology
- Geographic population
- Demographic representation
- Original consent records
- Licensing conditions
- Prediction timestamps

These limitations must be addressed before any real-world
deployment.

---

## 17. Final Decision on ML Justification

Machine learning is justified for this task as an educational
demonstration of binary classification and responsible ML
problem framing.

However, the supplied 12-record dataset is not sufficient to
justify production deployment.

A larger, representative and properly documented dataset would be
required before using the model for real customer decisions.

## Conclusion

This experiment framed customer churn prediction as a binary classification problem using customer-level historical information.

A majority-class non-ML baseline was established first. The majority class was "Not Churned" (0), giving an accuracy of 58.33% when predicting the majority class for all records.

A Logistic Regression model was then implemented using numerical preprocessing, categorical encoding, and feature scaling. The model achieved 100% accuracy, precision, recall, and F1-score on the three-record test set.

However, these results should not be interpreted as evidence of a highly accurate or production-ready model. The dataset contains only 12 records, and the test set contains only three records. Therefore, the reported metrics have high uncertainty and may change substantially with different samples.

The experiment demonstrates the complete ML problem-framing workflow, including defining the prediction target, establishing a baseline, preprocessing data, training a model, evaluating performance, identifying risks, and documenting responsible-use limitations.

The model should therefore be treated as an educational prototype rather than a system for making automated customer retention decisions.