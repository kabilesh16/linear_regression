import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Plot the data points
def plot_data(x, y):
    plt.scatter(x, y, color='blue', label='Data points')
    plt.title('Data Plot')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.show()

# Train the regression model
def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

# Estimate the y value for a given x input
def predict_value(model, x_value):
    return model.predict([[x_value]])[0]  # Directly return the scalar

# Main function
def main():
    # Load the CSV file (Ensure the file has columns labeled 'x' and 'y')
    file_path = 'data.csv'  # Replace with your CSV file path
    data = load_data(file_path)

    # Split data into X and Y
    X = data[['x']].values
    Y = data['y'].values

    # Plot the data
    plot_data(X, Y)

    # Train the model on the data
    model = train_model(X, Y)

    # Allow user to input a new x value
    try:
        x_input = float(input("Enter an X value to predict the corresponding Y value: "))
        y_pred = predict_value(model, x_input)
        print(f"Estimated Y value for X = {x_input}: {y_pred:.2f}")
    except ValueError:
        print("Invalid input. Please enter a numerical value for X.")

if __name__ == "__main__":
    main()
