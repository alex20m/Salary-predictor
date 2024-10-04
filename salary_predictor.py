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

def heatmap(data):
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def polynomialRegression():
    train_data = pd.read_csv('training_data.csv')
    val_data = pd.read_csv('validating_data.csv')
    test_data = pd.read_csv('testing_data.csv')

    #Extract the features (age, gender, education_level) and label (income)
    X_train = train_data[['age', 'gender', 'education_level']].values
    y_train = train_data['income'].values
    
    X_val = val_data[['age', 'gender', 'education_level']].values
    y_val = val_data['income'].values
    
    X_test = test_data[['age', 'gender', 'education_level']].values
    y_test = test_data['income'].values



if __name__ == "__main__":
    processed_data = pd.read_csv('processed_salary_data.csv')
    heatmap(processed_data)
    #polynomialRegression()