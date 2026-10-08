# ML Risk Register

## RabTech Academy - AI & Machine Learning Internship

| ID | Risk | Potential Impact | Mitigation / Safeguard | Monitoring |
|---|---|---|---|---|
| R01 | Very small dataset | Unreliable and unstable model performance | Use the model only as an educational baseline and collect a larger dataset before production use | Dataset size and validation performance |
| R02 | Unknown data provenance | Permission and ethical-use uncertainty | Document original source, permission and intended use | Periodic documentation review |
| R03 | Data leakage | Artificially high model performance | Use only information available at prediction time and separate train/test preprocessing | Leakage checks |
| R04 | False negatives | Genuine churners may not be identified | Monitor recall for the churned class | Recall and false-negative rate |
| R05 | False positives | Unnecessary retention outreach and cost | Monitor precision and define an intervention budget | Precision and false-positive rate |
| R06 | Distribution shift | Model performance may degrade when customer behavior changes | Monitor feature and prediction distributions | Distribution monitoring |
| R07 | Poor calibration | Predicted probabilities may not represent actual risk | Perform calibration evaluation before probability-based decisions | Calibration metrics |
| R08 | Limited subgroup information | Fairness differences cannot be adequately evaluated | Obtain approved subgroup information where appropriate and perform subgroup evaluation | Fairness metrics |
| R09 | Automated decision misuse | Model prediction may be treated as a certain outcome | Keep human review for consequential decisions | Human-review records |
| R10 | Input data quality problems | Invalid or unreliable predictions | Validate input data before prediction | Missingness and validation checks |
| R11 | Model degradation | Increasing prediction errors over time | Define performance thresholds and retraining/review procedures | Periodic model evaluation |
| R12 | Incorrect use outside intended purpose | Unexpected or harmful use of predictions | Clearly document intended and unsupported uses | Usage review |