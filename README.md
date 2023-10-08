# Real Estate Price Prediction using Support Vector Regression (SVR)
This repository contains a Python script for predicting house prices based on various features using Support Vector Regression (SVR). SVR is a machine learning algorithm that works well for regression tasks, especially when dealing with complex datasets.
# Dataset
The dataset used in this project is loaded from a CSV file named "Real estate.csv". It contains the following columns:
X3 distance to the nearest MRT station: Distance to the nearest Mass Rapid Transit station (in meters).
X4 number of convenience stores: Number of convenience stores in the vicinity.
Y house price of unit area: House price per unit area (in 10000 New Taiwanese Dollar/Ping).
# Data Preprocessing
Removed irrelevant columns: 'No', 'X1 transaction date', 'X2 house age'.
Checked for and handled missing values.
Checked for and removed outliers using the Interquartile Range (IQR) method.
# Feature Selection
Used Variance Inflation Factor (VIF) to identify and remove multicollinear features.
Dropped columns 'X5 latitude' and 'X6 longitude' based on VIF analysis.
# Model Building
Split the data into training and testing sets (80% training, 20% testing).
Implemented a base SVR model and evaluated its performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.
# Hyperparameter Tuning
Conducted a Randomized Search to find the best hyperparameters for the SVR model.
Optimized the SVR model using the best hyperparameters obtained from the Randomized Search.
# Single Variable Analysis
Analyzed the impact of individual variables on the prediction performance by training SVR models with single input features.
Compared R2 scores for both training and testing sets.
# Results
The final SVR model with optimized hyperparameters achieved the best performance with an R2 score of 65% on the test data.
Single variable analysis revealed that the 'X3 distance to the nearest MRT station' feature has the most significant impact on the prediction accuracy, followed by 'X4 number of convenience stores'.
