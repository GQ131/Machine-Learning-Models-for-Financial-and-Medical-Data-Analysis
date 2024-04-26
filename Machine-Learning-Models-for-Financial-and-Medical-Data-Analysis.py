from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np 
import pandas as pd 

fundamentals=pd.read_csv('fundamentals.csv')

#Drop columns: 
fundamentals.drop(['Unnamed: 0', 'Ticker Symbol', 'Period Ending', 'For Year'], axis=1, inplace=True)


fundamentals.dtypes

fundamentals.info()

fundamentals.columns

pd.set_option('display.max_rows', 80)
col_percentage = 100*fundamentals.isnull().sum()/len(fundamentals)
print(col_percentage)

# I fill out the missing values. 
# Calculate the median for ratios
median_cash_ratio = fundamentals['Cash Ratio'].median()
median_current_ratio = fundamentals['Current Ratio'].median()
median_quick_ratio = fundamentals['Quick Ratio'].median()

# Fill NaN for liquidity ratios with median
fundamentals['Cash Ratio'].fillna(median_cash_ratio, inplace=True)
fundamentals['Current Ratio'].fillna(median_current_ratio, inplace=True)
fundamentals['Quick Ratio'].fillna(median_quick_ratio, inplace=True)

# Fill For EPS and Estimated Shares Outstanding with median
median_eps = fundamentals['Earnings Per Share'].median()
median_estimated_shares = fundamentals['Estimated Shares Outstanding'].median()

fundamentals['Earnings Per Share'].fillna(median_eps, inplace=True)
fundamentals['Estimated Shares Outstanding'].fillna(median_estimated_shares, inplace=True)


train, test=train_test_split(fundamentals, train_size=0.7, random_state=4761)

from sklearn.metrics import mean_squared_error
train=train.dropna()
test=test.dropna()

X_train=train.drop(['Estimated Shares Outstanding'], axis=1)
y_train=train['Estimated Shares Outstanding'].values

X_test=test.drop(['Estimated Shares Outstanding'], axis=1)
y_test=test['Estimated Shares Outstanding'].values

rf_regressor = RandomForestRegressor() 

rf_regressor.fit(X_train, y_train)

rf_predictions = rf_regressor.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)

print("Random Forest MSE:", rf_mse)

# Model training with min_samples_split set to 3
rf_regressor_min_samples = RandomForestRegressor(min_samples_split=3, random_state=4761)
rf_regressor_min_samples.fit(X_train, y_train)

# Prediction and evaluation for the model with min_samples_split=3
rf_predictions_min_samples = rf_regressor_min_samples.predict(X_test)
rf_mse_min_samples = mean_squared_error(y_test, rf_predictions_min_samples)

print("Random Forest MSE with min_samples_split=3:", rf_mse_min_samples)


# using MDI to compute the variable importance in the random forest model
importances_mdi = rf_regressor.feature_importances_

print(importances_mdi)  



#Use Permutation Feasture Importance to compute the variable importance

from sklearn.inspection import permutation_importance

result = permutation_importance(rf_regressor, X_test, y_test, n_repeats=10, random_state=42)
importances_pfi = result.importances_mean

print(importances_pfi)


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Scaling values:
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

lasso_cv = LassoCV(alphas=None, cv=5, random_state=0) # when I say 'alphas=none' I let sklearn specify the amount of alphas. 

# Fit the model
lasso_cv.fit(X_train_scaled, y_train)

# Best alpha
print(f"Best alpha: {lasso_cv.alpha_}")

y_pred = lasso_cv.predict(X_test_scaled)

print("Lasso coefficients:", lasso_cv.coef_)
print("Intercept:", lasso_cv.intercept_)

from sklearn.metrics import r2_score
# Correct MSE calculation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R^2:", r2)

cancer_df=pd.read_csv('Cancer.csv')

# import necessary packages
from sklearn.tree import DecisionTreeClassifier

#remove unnamed column and Id column
cancer_df = cancer_df.loc[:, ~cancer_df.columns.str.contains('^Unnamed')]
cancer_df = cancer_df.loc[:, ~cancer_df.columns.str.contains('^id')]
#get the type of each column
cancer_df.dtypes

# Transforming "Diagnosis" column into a numeric variable
cancer_df['diagnosis'] = cancer_df['diagnosis'].map({'M': 1, 'B': 0})


X = cancer_df.drop(columns=['diagnosis'])
y = cancer_df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Fit the classifier
#classifier = DecisionTreeClassifier(max_depth=3) 

#In order to answer question 8, I let the tree fully grow
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predictions
predictions = classifier.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print(f'The MSE for Regression Tree is: {mse}')


from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.subplots(figsize=(22, 12))
plot_tree(classifier, feature_names=X_train.columns, proportion=True, fontsize=10, filled=True)
plt.show()


