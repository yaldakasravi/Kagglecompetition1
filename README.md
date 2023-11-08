Introduction

The code in this repository is designed for a binary classification task. It uses the scikit-learn library to train and evaluate several machine learning models, including Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting. The best model is selected based on hyperparameter tuning and is used to make predictions on a test dataset.

Getting Started

Before running the code, make sure you have the required Python libraries installed, such as numpy, pandas, and scikit-learn. You can install these libraries using pip or conda. Additionally, you will need to have your training and test datasets in CSV format. Update the file paths accordingly in the code.

Code Structure

The code is structured as follows:

Import necessary libraries and modules.
Define parameters and hyperparameters, including learning rates, regularization strengths, and normalization methods.
Implement data normalization functions (norm_data) to preprocess the input data.
Create a custom LogisticRegressionFromScratch class to train a logistic regression model from scratch.
Load and clean the training and test data.
Split the training data into training and validation sets.
Perform hyperparameter tuning using nested loops to find the best combination of hyperparameters.
Train machine learning models (Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting) and evaluate their performance.
Select the best-performing model based on validation accuracy.
Make predictions on the test data using the best model.
Save the predictions to a CSV file for submission.
Data Preparation

The code assumes that you have training and test data in CSV format. It loads and cleans the data, including removing unnecessary columns and performing data normalization. You can customize the data cleaning process according to your dataset.

Hyperparameter Tuning

The script includes a hyperparameter tuning section that allows you to search for the best hyperparameters for the selected model. You can specify a range of learning rates, regularization strengths, and the number of iterations to be considered during the search.

Model Training

The code trains several machine learning models with the specified hyperparameters. You can choose between implementing logistic regression from scratch or using scikit-learn's pre-built models. The best model is selected based on its performance on the validation dataset.

Model Evaluation

The code evaluates the selected model's performance on the validation dataset and prints the accuracy. It also provides counts of predicted labels and the distribution of true labels for further analysis.

Submission

Once the best model is selected, it is used to make predictions on the test dataset. The predictions are saved to a CSV file named "submission.csv" in the required format for submission.
