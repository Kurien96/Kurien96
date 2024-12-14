import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import tkinter as tk
from tkinter import messagebox

# Step 1: Data Collection and Preparation
print("Loading data...")
data = pd.read_csv('c:\\Users\\kkkos\\Desktop\\car.csv')
print("Data loaded successfully.")

print("Checking for missing values...")
print(data.isnull().sum())

print("Filling missing values for numeric columns...")
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

print("Filling missing values for categorical columns...")
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

print("Correcting inconsistencies...")
data['Fuel_Type'] = data['Fuel_Type'].str.lower()
data['Transmission'] = data['Transmission'].str.lower()

print("Handling outliers...")
data = data[data['Kms_Driven'] < 1000000]

print("Deriving new features...")
data['Car_Age'] = 2024 - data['Year']

print("Encoding categorical variables...")
data = pd.get_dummies(data, drop_first=True)

print("Selecting features and target variable...")
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

print("Normalizing/Standardizing numerical features...")
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Exploratory Data Analysis (EDA)
print("Performing EDA...")
print("Summary statistics:")
print(data.describe())

print("Saving histograms...")
data.hist(figsize=(10, 10))
plt.savefig('histograms.png')
plt.close()
print("Histograms saved...")

print("Saving boxplots...")
data.boxplot(figsize=(15, 10))
plt.xticks(rotation=90)
plt.savefig('boxplots.png')
plt.close()
print("Boxplots saved...")

print("Saving correlation heatmap...")
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.savefig('correlation_matrix.png')
plt.close()
print("Correlation heatmap saved...")

# Step 3 and 4: Model Development, Training, and Hyperparameter Tuning
print("Starting model development and hyperparameter tuning...")
model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Performing Grid Search with Cross-Validation...")
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

print(f'Best Hyperparameters: {grid_search.best_params_}')

print("Training model with best hyperparameters...")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

print("Making predictions on the test set...")
y_pred = best_model.predict(X_test)

print("Evaluating the model...")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# Step 5: Model Deployment
print("Saving the model...")
joblib.dump(best_model, 'car_price_predictor.pkl')

print("Loading the saved model...")
model = joblib.load('car_price_predictor.pkl')

# Tkinter GUI for Prediction
def predict_price():
    try:
        year = int(entry_year.get())
        present_price = float(entry_present_price.get())
        kms_driven = int(entry_kms_driven.get())
        fuel_type = fuel_var.get()
        seller_type = seller_var.get()
        transmission = transmission_var.get()
        owner = int(entry_owner.get())

        input_data = {
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner
        }

        input_df = pd.DataFrame(input_data, index=[0])
        input_df['Car_Age'] = 2024 - input_df['Year']
        input_df['Fuel_Type'] = input_df['Fuel_Type'].str.lower()
        input_df['Transmission'] = input_df['Transmission'].str.lower()
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        prediction = model.predict(input_df)
        
        result_label.config(text=f'Predicted Selling Price: {prediction[0]:.2f} lakhs')

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Creating the Tkinter window
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("600x600")  # Set the window size

# Customizing the window background
background_color = "#f0f0f0"
root.configure(bg=background_color)

# Creating input fields with custom styling
label_style = {"font": ("Helvetica", 14), "bg": background_color}
entry_style = {"font": ("Helvetica", 14), "width": 20}

tk.Label(root, text="Year", **label_style).grid(row=0, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Present Price (in lakhs)", **label_style).grid(row=1, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Kms Driven", **label_style).grid(row=2, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Fuel Type", **label_style).grid(row=3, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Seller Type", **label_style).grid(row=4, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Transmission", **label_style).grid(row=5, column=0, padx=20, pady=10, sticky="e")
tk.Label(root, text="Owner", **label_style).grid(row=6, column=0, padx=20, pady=10, sticky="e")

entry_year = tk.Entry(root, **entry_style)
entry_present_price = tk.Entry(root, **entry_style)
entry_kms_driven = tk.Entry(root, **entry_style)
fuel_var = tk.StringVar(root)
fuel_var.set("Select")
fuel_menu = tk.OptionMenu(root, fuel_var, "Petrol", "Diesel", "CNG")
fuel_menu.config(width=17, font=("Helvetica", 14))
seller_var = tk.StringVar(root)
seller_var.set("Select")
seller_menu = tk.OptionMenu(root, seller_var, "Dealer", "Individual")
seller_menu.config(width=17, font=("Helvetica", 14))
transmission_var = tk.StringVar(root)
transmission_var.set("Select")
transmission_menu = tk.OptionMenu(root, transmission_var, "Manual", "Automatic")
transmission_menu.config(width=17, font=("Helvetica", 14))
entry_owner = tk.Entry(root, **entry_style)

entry_year.grid(row=0, column=1, padx=20, pady=10)
entry_present_price.grid(row=1, column=1, padx=20, pady=10)
entry_kms_driven.grid(row=2, column=1, padx=20, pady=10)
fuel_menu.grid(row=3, column=1, padx=20, pady=10)
seller_menu.grid(row=4, column=1, padx=20, pady=10)
transmission_menu.grid(row=5, column=1, padx=20, pady=10)
entry_owner.grid(row=6, column=1, padx=20, pady=10)

# Creating the predict button with custom styling
predict_button = tk.Button(root, text="Predict Price", command=predict_price, font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="black")
predict_button.grid(row=7, columnspan=2, pady=30)

# Label to show the result

result_label = tk.Label(root, text="")
result_label.grid(row=8, columnspan=2)

# Running the Tkinter event loop
root.mainloop()
