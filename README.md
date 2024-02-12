# Project
Mercedes-Benz Greener Manufacturing

Summary:

Mercedes-Benz aims to reduce the time its vehicles spend on the test bench, ensuring safety and reliability while minimizing carbon dioxide emissions. The project involves the application of advanced data science techniques to optimize the testing system, focusing on predicting the time it takes for a car to pass testing based on its features. This optimization will contribute to faster testing and lower environmental impact without compromising safety standards.

Project Steps:

1. Data Inspection and Cleaning:

1.1. Variance Check:
Identified and removed columns with zero variance to enhance model efficiency.

```python
# Identify and remove columns with zero variance
zero_variance_cols = train_df.columns[train_df.var() == 0]
train_df = train_df.drop(zero_variance_cols, axis=1)
test_df = test_df.drop(zero_variance_cols, axis=1)
```

1.2. Null and Unique Values:
Checked for null and unique values in both the training and testing datasets.

1.3. Label Encoding:
Applied label encoding to categorical columns.

```python
# Apply label encoding to categorical columns
for col in categorical_cols:
    if col in test_df.columns:
        test_df[col] = label_encoder.transform(test_df[col])
```

2. Dimensionality Reduction:

Performed dimensionality reduction using Principal Component Analysis (PCA) to retain 95% of the variance.

```python
# Perform PCA for dimensionality reduction
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_standardized)
```

3. XGBoost Model Training:

Trained an XGBoost regressor on the training set and evaluated its performance on the validation set using Mean Squared Error (MSE).

```python
# Create an XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model on the training set
xgb_reg.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = xgb_reg.predict(X_val)

# Evaluate the model on the validation set
mse = mean_squared_error(y_val, y_val_pred)
print(f'Mean Squared Error on Validation Set: {mse}')
```

4. Predictions on Test Set:

Finally, predicted the test set values using the trained XGBoost model.

```python
# Predict the test_df values
test_predictions = xgb_reg.predict(test_df_pca)
print("Predicted Values for test_df:\n", test_predictions)
```
