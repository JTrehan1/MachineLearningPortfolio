import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
numberIsFraud = transactions['isFraud'].value_counts()
print(numberIsFraud)
# 282 corresponding to fraudulent transactions. 2
numberIsFraud = transactions['isFraud'].sum()
# print(numberIsFraud)

# Summary statistics on amount column
transactionAmount = transactions['amount']
print(transactionAmount.describe())
plt.hist(transactions['amount'], bins=50)
plt.xlim(0, 1000000) 
plt.title("Distribution of Transaction Amount")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.show()

# Create isPayment field
transactions["isPayment"] = 0
transactions["isPayment"][transactions["type"].isin(["PAYMENT","DEBIT"])]=1

# Create isMovement field
transactions["isMovement"] = 0
transactions["isMovement"][transactions["type"].isin(["CASH_OUT", "TRANSFER"])] = 1

# Create accountDiff field
transactions["accountDiff"] = transactions["oldbalanceOrg"] - transactions["oldbalanceDest"]

# Create features and label variables
# label column - isFraud
features = ["amount", "isPayment", "isMovement", "accountDiff"]
X = transactions[features]
y= transactions["isFraud"]

# Split dataset
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Normalize the features variables
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# Fit the model to the training data
logisticRegression = LogisticRegression()
logisticRegression.fit(xTrain, yTrain)

# Score the model on the training data
print(logisticRegression.score(xTrain, yTrain))

# Score the model on the test data
print(logisticRegression.score(xTest, yTest))
# The score returned is the percentage of correct classifications, or the accuracy, and will be an indicator for the sucess of the model.

# Print the model coefficients
print(logisticRegression.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([101238.78, 1.0, 1.0, 54670.1])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2, transaction3, your_transaction), axis=0)

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
predictions=logisticRegression.predict(sample_transactions)
print(predictions)
# Show probabilities on the new transactions
predictions_probability = logisticRegression.predict_proba(sample_transactions)
print(predictions_probability)