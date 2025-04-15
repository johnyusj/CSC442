# Read In Data
import pandas as pd
import numpy as np

goalies_df = pd.read_csv('Goalies.csv', on_bad_lines='skip')
teams_df = pd.read_csv('Teams.csv')

# Merge datasets
merged_df = pd.merge(
    teams_df,
    goalies_df,
    on=['year', 'tmID'],
    how='inner'
)

# 1. Drop unnecessary columns
columns_to_drop = ['lgID_y', 'confID', 'divID', 'franchID', 'name', 
                  'PostGP', 'PostMin', 'PostW', 'PostL', 'PostT', 'PostENG', 
                  'PostSHO', 'PostGA', 'PostSA', 'playoff', 'rank', 'ENG', 
                  'lgID_x', 'GA_y', 'W_x', 'L_x', 'T', 'OTL', 'W_y', 'L_y', 
                  'T/OL', 'SoW', 'SoL']
cleaned_df = merged_df.drop(columns=columns_to_drop)

# 2. Rename columns for clarity
cleaned_df = cleaned_df.rename(columns={'GA_x': 'GA'})

# 3. Group by team and year to get team-level statistics
team_stats = cleaned_df.groupby(['year', 'tmID']).agg({
    'G': 'first',          # Games played by team
    'Pts': 'first',        # Team points
    'GF': 'first',         # Goals for
    'GA': 'first',         # Goals against
    'GP': 'sum',           # Total games played by goalies
    'Min': 'sum',          # Total minutes by goalies
    'SHO': 'sum',          # Total shutouts
    'SA': 'sum',           # Total shots against
    'PIM': 'first',        # Penalties in minutes
    'PPG': 'first',        # Power play goals
    'PPC': 'first',        # Power play chances
    'PKG': 'first',        # Penalty kill goals against
    'PKC': 'first'         # Penalty kill chances
}).reset_index()

# 4. Calculate additional metrics
team_stats['SavePercentage'] = (team_stats['SA'] - team_stats['GA']) / team_stats['SA']
team_stats['GoalsAgainstAvg'] = (team_stats['GA'] * 60) / team_stats['Min']

# 5. Handle missing/infinite values
team_stats = team_stats.replace([np.inf, -np.inf], np.nan)

# 6. Display summary statistics and check for missing values
print("\nDataset Shape:", team_stats.shape)
print("\nMissing Values:")
print(team_stats.isnull().sum())
print("\nSummary Statistics:")
print(team_stats.describe())

# Optional: Save cleaned dataset
# team_stats.to_csv('cleaned_hockey_stats.csv', index=False)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Prepare features and target
features_to_drop = ['Pts', 'year', 'tmID']
X = team_stats.drop(columns=features_to_drop)
y = team_stats['Pts']

# Clean NaN and infinite values before modeling
print(f"\nOriginal number of rows: {len(X)}")

# 1. Remove rows with NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan)
mask = ~X.isna().any(axis=1)  # Create mask of rows without NaN
X = X[mask]
y = y[mask]

print(f"Number of rows after removing NaN/infinite values: {len(X)}")
print(f"Removed {len(team_stats) - len(X)} rows")

# Now proceed with the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import Linear Regression
from sklearn.linear_model import LinearRegression

# Train both models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lr_model = LinearRegression()

rf_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# Make predictions for both models
rf_pred = rf_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)

# Calculate metrics for both models
rf_mse = mean_squared_error(y_test, rf_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

rf_r2 = r2_score(y_test, rf_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Print strength of each modeling method
print("\nRandom Forest Performance:")
print(f"Root Mean Squared Error: {np.sqrt(rf_mse):.2f} points")
print(f"R-squared Score: {rf_r2:.3f}")

print("\nLinear Regression Performance:")
print(f"Root Mean Squared Error: {np.sqrt(lr_mse):.2f} points")
print(f"R-squared Score: {lr_r2:.3f}")

# Create side-by-side plots
plt.figure(figsize=(15, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Random Forest: Actual vs Predicted Points')

# Linear Regression plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Linear Regression: Actual vs Predicted Points')

plt.tight_layout()
plt.show()

# Feature importance for Linear Regression
lr_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_
})
lr_importance['abs_coefficient'] = abs(lr_importance['coefficient'])
lr_importance = lr_importance.sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features (Linear Regression):")
print(lr_importance.head(10))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Optional: Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print(f"Average CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
