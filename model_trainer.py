# --- 1) IMPORTING PACKAGES
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

#--- 2) DEFINE GLOBAL CONSTANTS
K = 10  # you can define the number of folds for cross-validation
SPLIT = 0.7  # define the ratio for train_test_split

#-- 3) DEFINE FUNCTIONS

#Define load_data() function
def load_data(path: str):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path: str, relative path of the CSV file

    :return     df: pd.DataFrame
    """
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

#Define create_target_and_predictors() function
def create_target_and_predictors(data: pd.DataFrame, target: str):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str, target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

#Define train_algorithm_with_cross_validation() function
def train_algorithm_with_cross_validation(X: pd.DataFrame, y: pd.Series):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable
    """
    accuracy = []
    for fold in range(0, K):
        model = RandomForestRegressor()
        scaler = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        trained_model = model.fit(X_train, y_train)

        y_pred = trained_model.predict(X_test)

        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

 #--- 4) MAIN FUNCTION
def main():
    # Load the data
    df = load_data("/path/to/your/data.csv")

    # Create predictors and target
    X, y = create_target_and_predictors(df, "estimated_stock_pct")

    # Train and evaluate the model
    train_algorithm_with_cross_validation(X, y)

if __name__ == "__main__":
    main()