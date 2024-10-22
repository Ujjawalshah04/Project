# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Collection
# For demonstration purposes, we'll generate synthetic crime data using pandas and numpy
# Replace this with actual historical crime data for real-world usage
data = pd.DataFrame({
    'Year': np.arange(2000, 2021),
    'Population': np.random.randint(100000, 500000, size=21),
    'Unemployment_Rate': np.random.uniform(3, 12, size=21),
    'Crime_Rate': np.random.uniform(200, 700, size=21)
})

# Step 2: Data Preprocessing
# Handling missing values (there are none in this synthetic data)
data.isnull().sum()

# Feature engineering: adding new feature (Crime Rate per 1000 people)
data['Crime_per_1000'] = data['Crime_Rate'] / data['Population'] * 1000

# Visualizing initial data (EDA)
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Year', y='Crime_Rate', marker="o")
plt.title('Crime Rate Over the Years')
plt.show()

# Step 3: Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Variables')
plt.show()

# Step 4: Model Selection
# Features and target variable
X = data[['Population', 'Unemployment_Rate']]
y = data['Crime_Rate']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Model Training (Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Step 6: Model Training (Random Forest Regression)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Model Evaluation (Linear Regression)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Step 8: Model Evaluation (Random Forest Regression)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Displaying Results
print(f"Linear Regression - MSE: {mse_lr}, MAE: {mae_lr}, R2 Score: {r2_lr}")
print(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R2 Score: {r2_rf}")

# Step 9: Predicting Future Crime Rates (e.g., for 2025)
future_data = pd.DataFrame({
    'Population': [450000],
    'Unemployment_Rate': [6.5]
})

predicted_crime_rate_lr = lr_model.predict(future_data)
predicted_crime_rate_rf = rf_model.predict(future_data)

print(f"Predicted Crime Rate (Linear Regression) for 2025: {predicted_crime_rate_lr[0]}")
print(f"Predicted Crime Rate (Random Forest) for 2025: {predicted_crime_rate_rf[0]}")
