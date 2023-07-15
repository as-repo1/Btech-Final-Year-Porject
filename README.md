# Btech-Final-Year-Porject

# Project Description

## Importing Libraries
The code begins by importing the necessary libraries such as pandas, matplotlib, and numpy.

## Reading and Preparing the Data
The code reads a CSV file named "WA_Fn-UseC_-Telco-Customer-Churn.csv" using pandas' `read_csv()` function and stores it in a DataFrame called `df`. The code then performs various operations on the DataFrame:
- Dropping the 'customerID' column using `df.drop()`.
- Checking the data types of the columns using `df.dtypes`.
- Converting the 'TotalCharges' column to numeric values using `pd.to_numeric()`.
- Identifying any null values in the 'TotalCharges' column.
- Filtering the DataFrame to exclude rows where 'TotalCharges' is null.
- Converting the 'TotalCharges' column to numeric values again after removing null values.

## Data Visualization
The code uses matplotlib to create visualizations based on the churn prediction data. It creates two histograms:
- The first histogram visualizes the distribution of the 'tenure' feature for customers with churn ('Churn=Yes') and without churn ('Churn=No').
- The second histogram visualizes the distribution of the 'MonthlyCharges' feature for customers with churn and without churn.

## Data Preprocessing
- The code defines a function called `print_unique_col_values(df)` to print unique values for object-type columns in the DataFrame.
- It replaces specific values in certain columns, such as 'No internet service' with 'No', using `df1.replace()`.
- It converts binary categorical columns ('Yes'/'No') to numeric values (1/0) using `df1[col].replace()` in a loop.
- It converts the 'gender' column values ('Female'/'Male') to numeric values using `df1['gender'].replace()`.

## One-Hot Encoding
The code uses pandas' `get_dummies()` function to perform one-hot encoding on categorical columns: 'InternetService', 'Contract', and 'PaymentMethod'. This process creates new columns representing the categories.

## Feature Selection
The code applies feature selection techniques to identify the top 20 features related to the target variable ('Churn'). It uses `SelectKBest` with chi-squared scoring to select the best features.

## Model Training and Evaluation
The code splits the data into training and testing sets using `train_test_split()`. It then builds and trains a neural network model using Keras. The model consists of multiple dense layers with ReLU activation and a sigmoid output layer. It is compiled with the Adam optimizer and binary cross-entropy loss. The model's training history is stored in the `history` variable.

## Model Performance Evaluation
The code evaluates the trained model on the test set using `model.evaluate()`. It also predicts the churn values for the test set using `model.predict()`. Then, it converts the predicted probabilities into binary predictions based on a threshold of 0.5. Several evaluation metrics such as accuracy, precision, recall, F1-score, and a confusion matrix are calculated and printed using sklearn's classification metrics.

## Additional Calculations
The code performs some additional calculations related to the confusion matrix and churn rate, but without further context, it's not clear what these calculations are intended for.

