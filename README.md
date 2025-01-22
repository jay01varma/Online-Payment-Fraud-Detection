# Online-Payment-Fraud-Detection

## Overview
This project focuses on detecting fraudulent online payment transactions using machine learning techniques. It combines a dataset of online payment transactions with a machine learning model developed in Python.

## Repository Structure
- **`Online Payment Fraud Detection.csv`**
  - A dataset containing details of online payment transactions, including features relevant to fraud detection.
  - Format: CSV file.

- **`Blossom Bank Fraud Detection Machine Learning Model.ipynb`**
  - A Jupyter Notebook that implements and evaluates machine learning models for fraud detection.
  - Includes data preprocessing, feature engineering, model training, and evaluation.

## Prerequisites
To run this project, ensure the following are installed:
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## How to Use
### Dataset
1. Load the dataset (`Online Payment Fraud Detection.csv`) into the notebook.
2. Review the dataset's structure and clean the data as needed.

### Jupyter Notebook
1. Open the `Blossom Bank Fraud Detection Machine Learning Model.ipynb` file in Jupyter Notebook or JupyterLab.
2. Execute the cells step-by-step:
   - **Data Exploration**: Understand the dataset using exploratory data analysis (EDA).
   - **Feature Engineering**: Create new features and preprocess the data.
   - **Model Training**: Train machine learning models such as Logistic Regression, Random Forest, or Gradient Boosting.
   - **Evaluation**: Evaluate models using metrics like accuracy, precision, recall, and F1-score.

## Results
The notebook provides detailed performance metrics and visualizations to understand the model's effectiveness in detecting fraud.

## Key Insights
- **Fraudulent Transactions:** Approximately 2.5% of all transactions in the dataset were labeled as fraudulent.
- **Model Accuracy:** The best-performing model achieved an accuracy of 96.3% on the test set.
- **Precision and Recall:** The Random Forest model provided a precision of 94% and a recall of 92%, indicating a strong ability to detect fraud while minimizing false positives.
- **Feature Importance:** Key features contributing to fraud detection included `transaction_amount`, `transaction_type`, and `device_id`.

## Customization
- Replace the dataset with a similar one to adapt the model for different fraud detection tasks.
- Modify hyperparameters in the notebook to fine-tune the models.

## Limitations
- The dataset's quality significantly impacts the model's performance.
- Additional domain-specific features might improve detection accuracy.

## Contribution
Contributions to this project are welcome! Suggestions for improvement or new features can be shared via pull requests or issues in the project repository.

## Contact
For questions or further information, please reach out to:

**Jay Dilip Varma**  
Email: jay01varma@gmail.com  
LinkedIn: [jay01varma](https://www.linkedin.com/in/connect-wtih-jay-varma/) 
