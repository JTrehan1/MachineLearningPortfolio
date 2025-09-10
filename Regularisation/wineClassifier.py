import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\jai.trehan\Desktop\Codeacadmey ML Career Path\Personal Git\MachineLearningPortfolio\Regularisation\wine_quality.csv")
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
# Transform feature datapoints to be on the same scale 
standard_scaler = StandardScaler()
standard_scaler.fit(features)
X = standard_scaler.transform(features)

## 2. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg = LogisticRegression(penalty=None)
clf_no_reg.fit(X_train, y_train)
## 4. Plot the coefficients
predictors = features.columns
coefficients = pd.Series(clf_no_reg.coef_.ravel(), predictors).sort_values()
coefficients.plot(kind='bar', title = "Coefficients (No regularisation)")
plt.show()

## 5. Training and test performance
from sklearn.metrics import f1_score
# Obtain training and test predictions
y_pred_train = clf_no_reg.predict(X_train)
y_pred_test = clf_no_reg.predict(X_test)

training_score = f1_score(y_train, y_pred_train)
testing_score = f1_score(y_test, y_pred_test)
print("Training Score", training_score)
print("Testing Score", testing_score)

## 6. Default Implementation (L2-regularized!)
clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)

# Plotting the coefficients with it L2 regularised 
coefficients_L2 = pd.Series(clf_default.coef_.ravel(), predictors).sort_values()
coefficients_L2.plot(kind='bar', title='Coefficients with L2 Regularisation')
plt.show()
plt.clf()
## 7. Ridge Scores
# Obtain model predictions 
y_pred_train_l2 = clf_default.predict(X_train)
y_pred_test_l2 = clf_default.predict(X_test)

training_score_l2 = f1_score(y_train, y_pred_train_l2)
testing_score_l2 = f1_score(y_test, y_pred_test_l2)
print("Training Score L2", training_score_l2)
print("Testing Score L2", testing_score_l2)

# The scores stayed the same - regularisation had no impact. The constraint boundary used is large enough to hold the orignal loss function minimum. To tune up regularisation, C values need to decrease. 

## 8. Coarse-grained hyperparameter tuning
training_array = []
testing_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]

for C_value in C_array:
  clf_default = LogisticRegression(C = C_value)
  clf_default.fit(X_train, y_train)
  
  y_pred_train_l2 = clf_default.predict(X_train)
  y_pred_test_l2 = clf_default.predict(X_test)

  training_score_l2 = f1_score(y_train, y_pred_train_l2)
  testing_score_l2 = f1_score(y_test, y_pred_test_l2)

  training_array.append(training_score_l2)
  testing_array.append(testing_score_l2)
  
print(f"Training Scores L2", training_array)
print(f"Testing Scores L2", testing_array)

## 9. Plot training and test scores as a function of C
plt.plot(C_array,training_array)
plt.plot(C_array,testing_array)
plt.title("Accuracy for Different C values")
plt.xlabel('C Values')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
C_array = np.logspace(-4, -2, 100)
tuning_C = [{'C': C_array}]

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
grid_search_Cv = GridSearchCV(estimator = clf_default, param_grid = tuning_C, scoring = 'f1', cv = 5, return_train_score = True)
grid_search_Cv.fit(X, y)

## 12. Optimal C value and the score corresponding to it
print("Best parameter", grid_search_Cv.best_params_)
print("Best score", grid_search_Cv.best_score_)

## 13. Validating the "best classifier"
clf_best_ridge = LogisticRegression(C = grid_search_Cv.best_params_['C'])
clf_best_ridge.fit(X_train, y_train)

y_pred_test_best = clf_best_ridge.predict(X_test)
testing_score_best = f1_score(y_test, y_pred_test_best)

print(f"Testing Scores L2", testing_score_best)

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
clf_l1 = LogisticRegressionCV(Cs = np.logspace(-2, 2, 100), cv =5, penalty = 'l1', solver = 'liblinear', scoring='f1')
clf_l1.fit(X, y)

## 15. Optimal C value and corresponding coefficients
print(clf_l1.C_)
print(clf_l1.coef_)

## 16. Plotting the tuned L1 coefficients
coef = pd.Series(clf_l1.coef_.ravel(),predictors).sort_values()

coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()