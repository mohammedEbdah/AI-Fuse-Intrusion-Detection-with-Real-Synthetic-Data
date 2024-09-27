import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import ipaddress

# Set the default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Define the paths to the folders
folder_paths = [
    "C:/Users/noori/Desktop/dataset/bengin",
    "C:/Users/noori/Desktop/dataset/compromised-ied",
    "C:/Users/noori/Desktop/dataset/compromised-scada",
    "C:/Users/noori/Desktop/dataset/external"
]

# Function to read and resample data from CSV files in the given folders
def load_and_resample_data(folder_paths, n_samples=50000):
    dataframes = []
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin-1')
                if len(df) > n_samples:
                    df = resample(df, n_samples=n_samples, random_state=123, replace=False)
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Function to check if a value is a valid IP address
def is_valid_ip(ip):
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

# Function to convert IP addresses to numerical format
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

# Load and resample data from the folders
network_data = load_and_resample_data(folder_paths)

# Drop 'No.' and 'Time' columns
network_data.drop(columns=['No.', 'Time'], inplace=True)

# Filter out non-IP addresses
network_data = network_data[network_data['Source'].apply(is_valid_ip)]
network_data = network_data[network_data['Destination'].apply(is_valid_ip)]

# Convert 'Source' and 'Destination' IP addresses to numerical format
network_data['Source'] = network_data['Source'].apply(ip_to_int)
network_data['Destination'] = network_data['Destination'].apply(ip_to_int)

# One-hot encode 'Protocol' column
network_data = pd.get_dummies(network_data, columns=['Protocol'])

# Replace 'Not set' with 0 and 'Set' with 1 in relevant columns
for column in ['Syn', 'Fin', 'Reset', 'Push', 'Urg', 'Ece', 'Cwr']:
    network_data[column] = network_data[column].map({'Not set': 0, 'Set': 1})

# Encode 'TCP Flags' hex values to integers
network_data['TCP Flags'] = network_data['TCP Flags'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

# Replace infinity values with NaN
network_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
network_data.dropna(inplace=True)

# Encode the column labels
label_encoder = LabelEncoder()
network_data['label'] = label_encoder.fit_transform(network_data['label'])

# Determine the number of unique classes
num_classes = network_data['label'].nunique()

# Split the data into training and testing datasets (80% train, 20% test)
train_dataset = network_data.sample(frac=0.8, random_state=42)
test_dataset = network_data.drop(train_dataset.index)

target_train = train_dataset['label']
target_test = test_dataset['label']

# Convert labels to categorical format
y_train = to_categorical(target_train, num_classes=num_classes)
y_test = to_categorical(target_test, num_classes=num_classes)

# Making train & test splits
X_train = train_dataset.drop(columns=['label']).values
X_test = test_dataset.drop(columns=['label']).values

# Convert data to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Reshape the data for CNN
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)

def create_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define the number of splits for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Split the data into train and validation sets
X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Convert fold data to float32
X_train_fold = X_train_fold.astype('float32')
X_val_fold = X_val_fold.astype('float32')
y_train_fold = y_train_fold.astype('float32')
y_val_fold = y_val_fold.astype('float32')

# Iterate through each fold
for fold in range(n_splits):
    model = create_model()
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)], verbose=0)

# After completing all folds, evaluate the final model on the testing set
final_scores = model.evaluate(X_test, y_test, verbose=0)
print("Final Testing Accuracy:", final_scores[1])
