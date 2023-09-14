import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report,confusion_matrix
from sklearn.linear_model import SGDClassifier
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

#Modelling
logging.basicConfig(filename='part2.log', level=logging.INFO)
# Define different parameter combinations to test
penalties = [None, 'l2', 'l1']
max_iters = [1000, 1500, 2000]
tols = [1e-3, 1e-4, 1e-5]


# Loop over parameter combinations
for penalty in penalties:
    for max_iter in max_iters:
        for tol in tols:
            # Creating and training the model with the current parameter combination
            model = SGDClassifier(penalty=penalty, max_iter=max_iter, tol=tol)
            model.fit(x_train, y_train)
            
            # Predicting the values for the test set
            y_pred = model.predict(x_test)
            
            # Calculating the training and testing accuracies
            training_accuracy = model.score(x_train, y_train)
            testing_accuracy = model.score(x_test, y_test)

            # Calculating the mean squared error
            mse = mean_squared_error(y_test, y_pred)
            
            # Logging the parameters and performance metrics
            logging.info(f'SGD Classifier parameters: penalty={penalty}, max_iter={max_iter}, tol={tol}')
            logging.info(f'Training accuracy: {training_accuracy}')
            logging.info(f'Testing accuracy: {testing_accuracy}')
            logging.info(f'MSE: {mse}')
            logging.info(f'Classification report:\n{classification_report(y_test, y_pred)}')
            logging.info(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')
            
            # Print to console for immediate feedback
            print(f'SGD Classifier parameters: penalty={penalty}, max_iter={max_iter}, tol={tol}')
            print(f'Training accuracy: {training_accuracy}')
            print(f'Testing accuracy: {testing_accuracy}')
            print(f'MSE: {mse}')
            print(f'Classification report:\n{classification_report(y_test, y_pred)}')
            print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')