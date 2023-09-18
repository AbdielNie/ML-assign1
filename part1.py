iemport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import logging

warnings.filterwarnings('ignore')

# import data
wine_dataset = pd.read_csv('https://raw.githubusercontent.com/AbdielNie/ML-assign1/main/winequality-red.csv',sep=";")
wine_dataset.head()
print(f'There are {wine_dataset.shape[0]} rows and {wine_dataset.shape[1]} columns')

#look into data
print(wine_dataset.sample(10, random_state=999))
print(wine_dataset.columns)
print(wine_dataset.isnull().sum())

#preprocess data
print(f"Check duplications: ",wine_dataset.duplicated())
print(f"Number of duplications ",wine_dataset.duplicated().sum())
print(f"Remove duplicated values and new dataset called 'df' ")
winedata = wine_dataset.drop_duplicates(inplace=False)
print(f"The shape of the data after droping duplication {winedata.shape}")
print(winedata.describe())
winedata['quality'] = winedata['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',
                                      6: 'good', 7: 'good', 8: 'good'})            
print(winedata['quality'].value_counts())
le = LabelEncoder()
winedata['quality'] = le.fit_transform(winedata['quality'])
print(winedata['quality'].value_counts)

#split data
x = winedata.iloc[:,:11]
y = winedata.iloc[:,11]

# determining the shape of x and y.
print("x.shape: ",x.shape)
print("y.shape: ",y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 44)

# determining the shapes of training and testing sets
print("x_train.shape: ",x_train.shape)
print("y_train.shape: ",y_train.shape)
print("x_test.shape: ",x_test.shape)
print("y_test.shape: ",y_test.shape)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#SGD Modelling

# Set up logging
logging.basicConfig(filename='part1.log', level=logging.INFO)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sgd_classifier(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        for i in range(m):
            random_idx = np.random.randint(m)
            xi = X[random_idx:random_idx+1]
            yi = y[random_idx:random_idx+1]
            gradient = xi.T.dot(sigmoid(xi.dot(theta)) - yi)
            theta -= learning_rate * gradient
        if epoch % 100 == 0:
            y_pred = sigmoid(X.dot(theta))
            mse = mean_squared_error(y, y_pred)
            print(f'Epoch {epoch}, MSE: {mse}')
            logging.info(f'Epoch {epoch}, MSE: {mse}')
    return theta

# Generate synthetic data
X = np.random.rand(1087, 11)
y = (X[:, 0] + X[:, 1]) > 1

# Add intercept term
X = np.c_[np.ones((X.shape[0], 1)), X]

# Define different parameter combinations to test
learning_rates = [0.01, 0.1, 0.5]
epochs_list = [1000, 1500, 2000]

# Variables to keep track of the best settings
best_mse = np.inf
best_learning_rate = None
best_epochs = None
best_report = None

# Loop over parameter combinations
for learning_rate in learning_rates:
    for epochs in epochs_list:
        # Train the model with the current parameter combination
        theta = sgd_classifier(X, y, learning_rate=learning_rate, epochs=epochs)

        # Get predictions for the classification report
        y_pred = (sigmoid(X.dot(theta)) > 0.5).astype(int)

        # Get and log the classification report
        report = classification_report(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        logging.info(f'Trial with learning_rate={learning_rate}, epochs={epochs}, MSE={mse}')
        logging.info(f'Classification report:\n{report}')
        
        # Update the best settings if current MSE is lower than the best MSE seen so far
        if mse < best_mse:
            best_mse = mse
            best_learning_rate = learning_rate
            best_epochs = epochs
            best_report = report

# Print and log the best settings found
print(f'Best settings found: learning_rate={best_learning_rate}, epochs={best_epochs}, MSE={best_mse}')
print(f'Classification report for best settings:\n{best_report}')
logging.info(f'Best settings found: learning_rate={best_learning_rate}, epochs={best_epochs}, MSE={best_mse}')
logging.info(f'Classification report for best settings:\n{best_report}')
