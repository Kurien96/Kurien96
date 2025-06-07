import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox

#  Training and testing datasets from CSV files
train_data_path = "C:\\Users\\kkkos\\Desktop\\Datasets\\House\\train.csv"  
test_data_path ="C:\\Users\\kkkos\\Desktop\\Datasets\\House\\test.csv"    

# Load the datasets into pandas DataFrames
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Combine both datasets into one (if you're training on both together)
df = pd.concat([train_df, test_df])

# Convert 'lot_size' from acre to sqft (if necessary)
df['lot_size'] = df.apply(lambda row: row['lot_size'] * 43560 if row['lot_size_units'] == 'acre' else row['lot_size'], axis=1)

# Handle missing values in 'lot_size' column by filling with the mean
df['lot_size'].fillna(df['lot_size'].mean(), inplace=True)

# Clean 'baths' column - handle decimal values
df['baths'] = df['baths'].round()  # Round decimals to nearest integer

# Clean 'beds' column - handle decimal values
df['beds'] = df['beds'].round()  # Round decimals to nearest integer

# Define features (X) and target variable (y)
X = df[['beds', 'baths', 'size', 'lot_size']]  # Features
y = df['price']  # Target variable

# Split the data into training and testing sets (using the same train/test split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Set up the Tkinter window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("600x400")  # Set a larger window size
root.config(bg="#f0f0f0")  # Background color for the window

# Function to predict house price
def predict_price():
    try:
        # Get input values from the user
        beds = int(beds_entry.get())
        baths = float(baths_entry.get())
        size = int(size_entry.get())
        lot_size = float(lot_size_entry.get())
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[beds, baths, size, lot_size]], columns=['beds', 'baths', 'size', 'lot_size'])
        
        # Predict the price
        predicted_price_usd = model.predict(input_data)[0]
        
        # Display the result in the label (in USD)
        result_label.config(text=f"Predicted House Price: ${predicted_price_usd:,.2f} USD", font=("Arial", 14, "bold"), fg="#4CAF50")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Create labels and entry fields for user input
tk.Label(root, text="Number of Bedrooms:", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, pady=10, padx=10, sticky="w")
beds_entry = tk.Entry(root, font=("Arial", 12), width=20)
beds_entry.grid(row=0, column=1, pady=10, padx=10)

tk.Label(root, text="Number of Bathrooms:", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, pady=10, padx=10, sticky="w")
baths_entry = tk.Entry(root, font=("Arial", 12), width=20)
baths_entry.grid(row=1, column=1, pady=10, padx=10)

tk.Label(root, text="House Size (sqft):", font=("Arial", 12), bg="#f0f0f0").grid(row=2, column=0, pady=10, padx=10, sticky="w")
size_entry = tk.Entry(root, font=("Arial", 12), width=20)
size_entry.grid(row=2, column=1, pady=10, padx=10)

tk.Label(root, text="Lot Size (sqft):", font=("Arial", 12), bg="#f0f0f0").grid(row=3, column=0, pady=10, padx=10, sticky="w")
lot_size_entry = tk.Entry(root, font=("Arial", 12), width=20)
lot_size_entry.grid(row=3, column=1, pady=10, padx=10)

# Button to trigger the prediction
predict_button = tk.Button(root, text="Predict Price", font=("Arial", 14), bg="#4CAF50", fg="white", command=predict_price, width=20, height=2)
predict_button.grid(row=4, columnspan=2, pady=20)

# Label to display the result
result_label = tk.Label(root, text="Predicted House Price: $0.00 USD", font=("Arial", 16), fg="#4CAF50", bg="#f0f0f0")
result_label.grid(row=5, columnspan=2)

# Start the Tkinter event loop
root.mainloop()
