import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

def load_data(folder_paths, sample_size=50000):
    """
    Load data from CSV files in multiple folders, drop 'No.' and 'Time' columns if they exist.
    """
    data = pd.DataFrame()
    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                # Drop 'No.' and 'Time' columns if they exist
                df = df.drop(columns=[col for col in ['No.', 'Time'] if col in df.columns])
                # Take a random sample of rows from each file
                df = df.sample(n=min(sample_size, df.shape[0]), random_state=42)
                data = pd.concat([data, df], axis=0)
    return data

def preprocess_data(data):
    """
    Preprocess data by converting categorical variables into numerical format using one-hot encoding.
    """
    return pd.get_dummies(data)

def main():
    # List of folder paths containing CSV files
    folder_paths = [
        "C:/Users/noori/Desktop/dataset/bengin", 
        "C:/Users/noori/Desktop/dataset/compromised-ied",
        "C:/Users/noori/Desktop/dataset/compromised-scada",
        "C:/Users/noori/Desktop/dataset/external"
    ]

    # Load data
    data = load_data(folder_paths)

    # Preprocess data
    data = preprocess_data(data)

    # Assuming the last column in the CSV files is the target variable
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for numerical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define parameter grid for grid search with KNN
    param_grid = {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Initialize KNN classifier within a pipeline with preprocessing
    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', KNeighborsClassifier())])

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=knn_pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # Predictions using the best model
    best_knn_pipeline = grid_search.best_estimator_
    y_pred = best_knn_pipeline.predict(X_test)

    # Evaluate accuracy of the 'label' column
    label_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the 'label' column:", label_accuracy)

if __name__ == "__main__":
    main()