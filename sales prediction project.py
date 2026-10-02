import pandas as pd

df = pd.read_csv("Advertising.csv")

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())
df = df.drop("Unnamed: 0", axis=1)

print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
import matplotlib.pyplot as plt

plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()

plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()

plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()

print("\nCorrelation with Sales:")
print(df.corr()["Sales"])

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Sales:")
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head(10))

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")

plt.show()

# Advertising Impact Analysis

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nAdvertising Impact:")
print(coefficients)

# Future Sales Prediction

new_ad = pd.DataFrame({
    "TV": [200],
    "Radio": [30],
    "Newspaper": [20]
})

predicted_sales = model.predict(new_ad)

print("\nFuture Sales Prediction:")
print("Predicted Sales:", predicted_sales[0])

# Comparing Advertising Scenarios

scenarios = pd.DataFrame({
    "TV": [100, 200, 250],
    "Radio": [20, 30, 40],
    "Newspaper": [10, 20, 30]
})

scenario_predictions = model.predict(scenarios)

scenarios["Predicted Sales"] = scenario_predictions

print("\nAdvertising Scenarios:")
print(scenarios)

# Step 14 - Actionable Business Insights

print("\n========== ACTIONABLE BUSINESS INSIGHTS ==========")

print("\n1. TV advertising has the strongest relationship with Sales.")

print("\n2. Radio advertising has a moderate positive relationship with Sales.")

print("\n3. Newspaper advertising has a weaker relationship with Sales.")

print("\n4. Predicted sales increase across the three advertising scenarios.")

print("\n5. The model can be used to estimate sales for different advertising combinations.")

print("\n6. The model should be used as a decision-support tool,")
print("   because other factors affecting sales are not included in the dataset.")


# Scenario Comparison Graph

plt.figure(figsize=(8, 5))

plt.bar(
    ["Scenario 1", "Scenario 2", "Scenario 3"],
    scenarios["Predicted Sales"]
)

plt.xlabel("Advertising Scenario")
plt.ylabel("Predicted Sales")
plt.title("Advertising Scenarios vs Predicted Sales")

plt.show()


# Final Project Summary

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Project: Sales Prediction Using Python")
print("Model: Linear Regression")
print("Features: TV, Radio, Newspaper")
print("Target: Sales")

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nBest Predicted Sales Scenario:")
print(scenarios.loc[
    scenarios["Predicted Sales"].idxmax()
])