# Car Price Prediction with Machine Learning using PyTorch

This project demonstrates how to predict car prices using a regression model built with PyTorch. It includes preprocessing of real-world car data, exploratory data analysis (EDA), feature engineering, model training, and result visualization.

📊 Insights from Data Analysis
Data Overview
Total rows: 10,000+

Key features: company, model, year, fuel_type, kms_driven, owner_type, mileage, engine, max_power, seats, price

Target variable: price

Data Cleaning and Preprocessing
Converted features like engine, max_power, and mileage to numeric formats.

Handled missing values across multiple features using imputation or removal.

One-hot encoded categorical variables (e.g., fuel_type, owner_type).

Feature Engineering
Extracted relevant numeric values from strings (e.g., "1498 CC" → 1498).

Applied Min-Max scaling to all numeric features.

Converted year of purchase into car age for better modeling.

Exploratory Data Analysis (EDA)
Correlation heatmap shows strong negative correlation between year and price, and strong positive correlation between engine, max_power, and price.

Box plots highlight outliers in features like price, engine, and max_power.

Distributions show that price, engine, and max_power are right-skewed.

Company-wise average price indicates premium brands (e.g., Mercedes, BMW, Audi) command higher average prices.

Modeling with PyTorch
Built a custom neural network using PyTorch with the following:

Input layer matching the number of processed features.

Hidden layers with ReLU activations.

Dropout for regularization.

Output layer with a single regression output (price).

Used MSELoss and Adam optimizer.

Trained for 500+ epochs with monitoring of loss and validation RMSE.

Model Evaluation
Training and validation losses converge smoothly.

Final validation RMSE: ₹114,924.93

The model generalizes well with low error margin considering the price range.

📈 Visualizations
Actual vs Predicted Price Plot: Shows strong alignment between real and predicted values with minor dispersion.

Residual Plot: Random scatter around zero indicating good fit and minimal bias.

Price Distribution Plot: Predicted prices follow a similar distribution to actual prices.

Feature Importance: Based on model weights, engine, max_power, and year have higher influence on price.
