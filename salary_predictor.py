import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
import seaborn as sns

# Regression import 

from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 

# Used to plot correlations in from the data
def heatmap(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def choose_degree(X_train, y_train, X_val, y_val):
    validation_errors = []
    degrees = range(1, 11)
    for degree in degrees:
        model = LinearRegression()
        poly = PolynomialFeatures(degree=degree)

        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        model.fit(X_train_poly, y_train)
        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        validation_errors.append(val_mse)        

    # Identify the best degree based on validation errors
    best_degree = degrees[np.argmin(validation_errors)]
    print(f'The best polynomial degree is: {best_degree}')


def polynomialRegression():
    train_data = pd.read_csv('training_data.csv')
    val_data = pd.read_csv('validating_data.csv')
    test_data = pd.read_csv('testing_data.csv')

    # Extract the features (age, gender, education_level) and label (income)
    X_train = train_data[['age', 'gender', 'education_level']].values
    y_train = train_data['income'].values
    
    X_val = val_data[['age', 'gender', 'education_level']].values
    y_val = val_data['income'].values
    
    X_test = test_data[['age', 'gender', 'education_level']].values
    y_test = test_data['income'].values

    #choose_degree(X_train, y_train, X_val, y_val)

    degree = 8  # Can be changed, 8 found to be the best using validation set

    model = LinearRegression()
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    model.fit(X_train_poly, y_train)

    y_test_pred = model.predict(X_test_poly)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f'Test MSE: {test_mse:.2f}')


if __name__ == "__main__":
    #processed_data = pd.read_csv('processed_salary_data.csv')
    #heatmap(processed_data)
    polynomialRegression()