import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def train_model(dataset, form_data):
    df = pd.read_csv(dataset.file.path)

    target_column = form_data['target_column']
    feature_columns = form_data['feature_columns']
    task = form_data['task']
    algorithm = form_data['algorithm']
    train_test_split_percentage = form_data['train_test_split']

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_test_split_percentage, random_state=42)

    model = None  # Set model to None by default

    if task == 'classification':
        if algorithm == 'dt':
            model = DecisionTreeClassifier()
        elif algorithm == 'rf':
            model = RandomForestClassifier()
    elif task == 'regression':
        if algorithm == 'lr':
            model = LinearRegression()
        elif algorithm == 'ridge':
            model = Ridge()

    if model is not None:  # Check if the model is not None before fitting
        model.fit(X_train, y_train)

        # You can save the model using a library like joblib or pickle, and store the model_id for future use.
        # For this example, we'll just return the model along with X_test and y_test for testing purposes.
        results = {
            'model': model,
            'X_test': pd.DataFrame(X_test, columns=feature_columns),
            'y_test': y_test
        }
        return results
    else:
        raise ValueError("The specified task and algorithm combination is not supported.")

def test_model(dataset, model_id):
    # In a real-world scenario, you would load the model using the model_id.
    # For this example, we'll assume the model is passed directly along with X_test and y_test.
    model = model_id['model']
    X_test = model_id['X_test']
    y_test = model_id['y_test']

    y_pred = model.predict(X_test)

    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {'accuracy': accuracy, 'f1_score': f1}
    elif isinstance(model, (DecisionTreeRegressor, RandomForestRegressor, LinearRegression, Ridge)):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mean_squared_error': mse, 'r2_score': r2}
