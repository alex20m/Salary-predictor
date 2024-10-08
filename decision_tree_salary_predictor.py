import numpy as np  # import numpy package under shorthand "np"
import pandas as pd  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
import seaborn as sns

# Tree import
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error  # function to calculate mean squared error





def decissionTreeRegressor():
    data = pd.read_csv('processed_salary_data.csv')
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

    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'criterion': ['squared_error', 'friedman_mse'],
        'splitter': ['best', 'random']
    }

    regressor = DecisionTreeRegressor(random_state=0)

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')


    grid_search.fit(X_train, y_train)

    best_regressor = grid_search.best_estimator_

    y_pred_train = best_regressor.predict(X_train)
    y_pred_test = best_regressor.predict(X_test)
    y_pred_val = best_regressor.predict(X_val)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_val = mean_squared_error(y_val, y_pred_val)

    #pd.DataFrame(y_pred).to_csv("predicted_salary.csv", index=False)

    print('Without GridSearchCv:\nValidation MSE: 151997896.78\nTest MSE: 81337107.86')

    print('With GridSearchCV:\n')
    print(f'Train MSE:    {mse_train:.2f}')
    print(f'Test  MSE:    {mse_test:.2f}')
    print(f'Validate MSE: {mse_val:.2f}')
    print(grid_search.best_params_)


if __name__ == "__main__":
    decissionTreeRegressor()