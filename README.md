**Problem Overview**

This project aims to predict whether a bank customer will subscribe to a term deposit using Logistic Regression. The dataset used is the Bank Marketing Dataset, which contains customer demographic and behavioral information.

**Dataset Description**

=> Total Attributes: 17

=> Target Variable: y (yes/no)

=> Features include:

   - Age

   - Job

   - Marital Status

   - Education

   - Balance

   - Contact type

   - Campaign details

**Methodology**

*Data Preprocessing*
  
  - Converted target variable (yes/no) to binary (1/0)
  - Encoded categorical variables using Label Encoding
  - Split dataset into training (80%) and testing (20%)

*Model Used*

  - Logistic Regression
  - Chosen because it is suitable for binary classification

**Evaluation Metrics**

  - Accuracy Score
  - Confusion Matrix
  - Precision, Recall, F1-score

**Findings**

  - Logistic Regression performs reasonably well on structured banking data
  - Model performance depends on feature encoding and data balance
  - Some features like "duration" strongly influence prediction
