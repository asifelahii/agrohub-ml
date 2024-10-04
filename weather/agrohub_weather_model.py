# agrohub_weather_model.py

# --------------------------
# Importing necessary libraries
# --------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------
# 1. Data Preprocessing
# --------------------------
def load_and_clean_data(weather_data):
    """Loads weather data, cleans, and handles missing values."""
    df = pd.read_csv("weather_data.csv")
    
    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')


    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    non_numeric_cols = df.select_dtypes(exclude='number').columns
    
    # Fill missing numeric values with the column mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing non-numeric values with the mode (most frequent value)
    for col in non_numeric_cols:
        if not df[col].mode().empty:  # Check if mode is not empty
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna('')  # Or choose any default value
    
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def add_features(weather_data):
    """Creates new features like average temperature and pressure difference."""
    # Creating new features
    weather_data['Avg_Temp'] = (weather_data['Temp9am'] + weather_data['Temp3pm']) / 2
    weather_data['Temp_Diff'] = weather_data['MaxTemp'] - weather_data['MinTemp']
    weather_data['Pressure_Diff'] = weather_data['Pressure3pm'] - weather_data['Pressure9am']
    
    # Check if 'RainTomorrow' exists in the dataset
    if 'RainToday' in weather_data.columns: # 'RainTomorrow' has been replaced by 'RainToday' in this whole section
        # Convert 'Yes'/'No' values to 1/0
        weather_data['RainToday'] = weather_data['RainToday'].map({'Yes': 1, 'No': 0})
    else:
        print("Error: 'RainTomorrow' column not found in the dataset.")
        print(f"Available columns: {weather_data.columns}")
        return None  # Exit the function if column is missing
    
    return weather_data

# --------------------------
# 2. Exploratory Data Analysis (EDA) Section
# --------------------------
def perform_eda(weather_data):
    """Performs basic EDA like correlation heatmap."""
    print("First 5 rows of the dataset:")
    print(weather_data.head())
    
    # Select only numeric columns for correlation
    numeric_cols = weather_data.select_dtypes(include='number')
    
    # Correlation Heatmap
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# --------------------------
# 3. Model Training Section
# --------------------------
def train_weather_model(weather_data):
    """Trains the Logistic Regression model."""
    # Define features and target
    X = weather_data[['Avg_Temp', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm']]
    y = weather_data['RainToday']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # Predictions and accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    
    # Save the trained model
    joblib.dump(model, 'weather_model.pkl')
    
    return model, X_test, y_test, y_pred

# --------------------------
# 4. Model Evaluation Section
# --------------------------
def evaluate_model(y_test, y_pred):
    """Evaluates the model performance using a confusion matrix."""
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# --------------------------
# Main Execution: Combining all sections
# --------------------------
if __name__ == "__main__":
    # Load and preprocess the data
    weather_data = load_and_clean_data('weather_data.csv')
    weather_data = add_features(weather_data)
    
    # Perform exploratory data analysis (optional)
    perform_eda(weather_data)
    
    # Train the model
    model, X_test, y_test, y_pred = train_weather_model(weather_data)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred)
    
    print("Model training and evaluation completed.")