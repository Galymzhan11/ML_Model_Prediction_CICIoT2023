# Network Traffic Analysis Using Machine Learning

## Overview

This project focuses on network traffic analysis using various machine learning algorithms to detect and classify cyber threats. The dataset used is CICIoT2023, which contains multiple types of network attacks, including DDoS, DNS spoofing, reconnaissance attacks, and more.

## Features

- Utilizes multiple machine learning algorithms for network intrusion detection.
- Implements feature engineering techniques to enhance model performance.
- Introduces noise into training labels to prevent overfitting.
- Evaluates models using various performance metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
- Supports data preprocessing, balancing, and transformation.

## Dataset

The dataset used is **CICIoT2023**, which contains:

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset)

- 169 CSV files totaling approximately 13 GB.
- 45.6 million rows of network traffic data.
- Multiple attack categories, including DoS, DDoS, Brute Force, Spoofing, and more.
- Boolean protocol indicators for HTTP, HTTPS, DNS, SSH, and others.

## Installation

```bash
# Clone the repository
git clone https://github.com/Galymzhan11/ML_Model_Prediction_CICIoT2023.git

cd ML_Model_Prediction_CICIoT2023

# Install dependencies
# Since there is no `requirements.txt` file, install dependencies manually:
# Example:
# pip install pandas scikit-learn numpy matplotlib (Optional) Install necessary dependencies
```

## Usage

```bash
# Run the dataset extraction script first
python DatasetExtraction.py

# Run the main script
python MainModels.py

```

## Machine Learning Models Used

- **Naive Bayes**
- **Random Forest**
- **Decision Tree**
- **AdaBoost**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **XGBoost**

## Model Performance Metrics

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Naive Bayes         | 0.86     | 0.92      | 0.77   | 0.84     | 0.85    |
| Random Forest       | 0.93     | 0.91      | 0.95   | 0.93     | 0.93    |
| Decision Tree       | 0.92     | 0.91      | 0.92   | 0.92     | 0.92    |
| AdaBoost            | 0.93     | 0.92      | 0.94   | 0.93     | 0.93    |
| KNN                 | 0.93     | 0.91      | 0.95   | 0.93     | 0.93    |
| Logistic Regression | 0.93     | 0.91      | 0.95   | 0.93     | 0.93    |
| XGBoost             | 0.93     | 0.91      | 0.95   | 0.93     | 0.93    |


## Future Improvements

- Integration with real-time network monitoring systems.
- Exploration of deep learning techniques for better accuracy.
- Further feature engineering to improve model efficiency.

## License

This project is open-source and available under the MIT License.
