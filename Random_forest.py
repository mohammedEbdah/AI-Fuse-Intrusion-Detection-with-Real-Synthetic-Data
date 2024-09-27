import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

def load_data(folder_paths, sample_size=50000):
    data = pd.DataFrame()
    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df = df.drop(columns=[col for col in ['No.', 'Time'] if col in df.columns])
                df = df.sample(n=min(sample_size, df.shape[0]), random_state=42)
                data = pd.concat([data, df], axis=0)
    return data

def preprocess_data(data):
    return pd.get_dummies(data)

def main():
    folder_paths = [
        "C:/Users/noori/Desktop/dataset/bengin",
        "C:/Users/noori/Desktop/dataset/compromised-ied",
        "C:/Users/noori/Desktop/dataset/compromised-scada",
        "C:/Users/noori/Desktop/dataset/external"
    ]

    data = load_data(folder_paths)
    data = preprocess_data(data)

    if data.isnull().values.any():
        data = data.dropna()

    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],  # Testing 50 and 100 boosting rounds
        'max_depth': [3, 10, 20],   # Different depths
        'learning_rate': [0.01, 0.1, 0.2],  # Different learning rates
        'subsample': [0.8, 1.0]     # Different subsample rates
    }

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, early_stopping_rounds=5)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the model using grid search
    grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy:", accuracy)

if __name__ == "__main__":
    main()
