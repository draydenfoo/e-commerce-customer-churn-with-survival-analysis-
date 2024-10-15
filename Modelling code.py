#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:14:16 2024

@author: draydenfoo
this code will have 2 df variables from same file. it is for using mac or windows. 
remember to # the one not using according to OS type you are using
"""

# check if packages installed. if not then will install
import subprocess
import sys
import platform

packages_to_install = [
    'numpy', 'pandas', 'scipy', 'lifelines', 
    'matplotlib', 'tensorflow', 
    'seaborn','scikit-learn']


# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install each package
for package in packages_to_install:
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. will install now ")
        install(package)
        print(f"{package} installed.")



#link to dataset
#https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis



#important libraries
import numpy as np # linear algebra
import pandas as pd # data processing, (e.g. pd.read_csv)
from scipy.stats import skew
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from keras.layers import Input
import datetime
import matplotlib.pylab as plt
import seaborn as sns #helpful for exploratory plots 
plt.style.use('ggplot')

# Create a filename
# Determine the platform and set the file location accordingly
if platform.system() == 'Windows':
    df=pd.read_csv("C:/Users/drayden foo/OneDrive - Sunway Education Group/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv", low_memory=False)
else:
    df=pd.read_csv("/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv")


#Data Cleaning process--------------------------------------------------------------------------------------------

#find number of missing values 
missing_values = df.isnull().sum()
print("Missing Values:")
missing_values

# Replace missing 'Returns' values with zero
no_return_value = 0
df['Returns'].fillna(0, inplace=True)
gender_mapping = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(gender_mapping)


# Convert 'Purchase Date' to datetime and sort data
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y %H:%M')
df = df.sort_values(by='Purchase Date')



# Group by customer and product category, then calculate the time between consecutive purchases
def calculate_durations(group):
    group['Duration Between Purchases'] = group['Purchase Date'].diff().dt.days.fillna(0)
    return group

# Apply the function to calculate durations for each customer and product category
df_with_durations = df.groupby(['Customer ID', 'Product Category']).apply(calculate_durations)

df_with_durations.reset_index(drop=True, inplace=True)

#find numeric variables:
numeric_var = df.select_dtypes(include=[np.number])
numeric_var.info()
numeric_summary = numeric_var.describe()
print(numeric_summary)
numeric_summary.to_csv('numeric data summary.csv')# did this to show summary. spyder console too small to see


#find categorical variables:
categorical_var = df.select_dtypes(include=['object', 'category', 'string'])
categorical_var.info()
categorical_summary = categorical_var.describe()
print(categorical_summary)
categorical_summary.to_csv('categorical data summary.csv')# did this to show summary. spyder console too small to see




#full data encoding
# Encoding 'Payment Method' column
df_encoded_payment = pd.get_dummies(df_with_durations['Payment Method'], prefix='Payment_Method', drop_first=True)
df_encoded_payment['Payment_Method_PayPal'] = df_encoded_payment['Payment_Method_PayPal'].astype(int)
df_encoded_payment['Payment_Method_Credit Card'] = df_encoded_payment['Payment_Method_Credit Card'].astype(int)

# Encoding 'Product Category' column
df_encoded_category  = pd.get_dummies(df_with_durations['Product Category'], prefix='Product_Category', drop_first=True)
df_encoded_category['Product_Category_Clothing'] = df_encoded_category['Product_Category_Clothing'].astype(int)
df_encoded_category['Product_Category_Electronics'] = df_encoded_category['Product_Category_Electronics'].astype(int)
df_encoded_category['Product_Category_Home'] = df_encoded_category['Product_Category_Home'].astype(int)

# Concatenating the encoded DataFrames with the original DataFrame
df_encoded_full = pd.concat([df_with_durations, df_encoded_payment, df_encoded_category], axis=1)

# Dropping the original categorical columns
df_encoded_full.drop(['Payment Method', 'Product Category', 'Purchase Date', 'Customer Name'], axis=1, inplace=True)
#remove unecessary df
del df_encoded_payment
del df_encoded_category
        

# Export the DF 
# Load the sample dataset
if platform.system() == 'Windows':
    df_encoded_full.to_excel('C:/Users/drayden foo/OneDrive - Sunway Education Group/Uni/20058798_Foo Wai Loon_Apr24/System Files/df_encoded_full.xlsx', index=False, sheet_name='Sheet1')
else:
    df_encoded_full.to_excel('/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files//df_encoded_full.xlsx', index=False, sheet_name='Sheet1')

'''Correlation matrix test'''
correlation_matrix = df_encoded_full.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

    



'''------------------------------------KMF modelling section------------------------------------'''
# General Kaplan-Meier Survival Analysis
kmf = KaplanMeierFitter()

# Fit the model with duration and status
kmf.fit(df_encoded_full['Duration Between Purchases'], event_observed=df_encoded_full['Churn'])
# Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot(ci_show=False, show_censors=True, censor_styles={'marker': '|', 'ms': 12})
plt.title("General Kaplan-Meier Survival Curve")
plt.xlabel("Days")  
plt.ylabel("Survival Probability")
plt.show()



''' survival KMF for gender'''
# Create separate subplots for males and females graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot for males
males = df_encoded_full[df_encoded_full['Gender'] == 0]
kmf.fit(males['Duration Between Purchases'], event_observed=males['Churn'], label="Male")
kmf.plot_survival_function(ax=ax1)
ax1.set_title("Survival of Males")
ax1.set_xlabel("Timeline")
ax1.set_ylabel("Survival Probability")


# Plot for females
females = df_encoded_full[df_encoded_full['Gender'] == 1]
kmf.fit(females['Duration Between Purchases'], event_observed=females['Churn'], label="Female")
kmf.plot_survival_function(ax=ax2)
ax2.set_title("Survival of Females")
ax2.set_xlabel("Timeline")
ax2.set_ylabel("Survival Probability")

plt.tight_layout()
plt.show()





'''product category KMF plot in 1 plot '''
def plot_kmf_survival_curves_one_hot(df, duration_col, event_col, one_hot_columns):
  
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(12, 8))

    for column in one_hot_columns:
        category_df = df[df[column] == 1]
        kmf.fit(category_df[duration_col], event_observed=category_df[event_col], label=column)
        kmf.plot(ci_show=False, show_censors=True, censor_styles={'marker': '|', 'ms': 12})

    plt.title(f"Kaplan-Meier Survival Curves by {column} Product Category")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.legend(title="Product Category")
    plt.show()
    
    # log-rank test
    print("Log-Rank Test Results:")
    for i in range(len(one_hot_columns)):
        for j in range(i + 1, len(one_hot_columns)):
            group1 = df[df[one_hot_columns[i]] == 1]
            group2 = df[df[one_hot_columns[j]] == 1]
            results = logrank_test(group1[duration_col], group2[duration_col], event_observed_A=group1[event_col], event_observed_B=group2[event_col])
            print(f"Comparison between {one_hot_columns[i]} and {one_hot_columns[j]}: p-value = {results.p_value:.4f}")

    for column in one_hot_columns:
        category_df = df[df[column] == 1]
        kmf.fit(category_df[duration_col], event_observed=category_df[event_col])
        survival_table = kmf.event_table
        print(f"Survival table for {column}:")
        print(survival_table[['at_risk']])
        print()

plot_kmf_survival_curves_one_hot(df_encoded_full, 'Duration Between Purchases', 'Churn', ['Product_Category_Electronics', 'Product_Category_Home', 'Product_Category_Clothing'])







'''------------------------------------CPH modelling section------------------------------------'''

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df_encoded_full, test_size=0.2, random_state=42)


#Check for proportionality
# Check the shape and columns of the DataFrame
print("Shape of DataFrame:", df_encoded_full.shape)
print("Columns in DataFrame:", df_encoded_full.columns)

# Split data for training and testing
train_data, test_data = train_test_split(df_encoded_full, test_size=0.2, random_state=42)

# Verify columns in train_data
print("Columns in train_data:", train_data.columns)

# Define the columns to be used in the model
#duration_col = 'Duration Between Purchases'
#event_col = 'Churn'
covariates = ['Payment_Method_PayPal', 'Payment_Method_Credit Card','Product_Category_Clothing', 'Product_Category_Electronics', 'Product_Category_Home']

# Ensure train_data includes only the necessary columns for fitting the model
train_data = train_data[['Duration Between Purchases', 'Churn'] + covariates]


# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
#cph.fit(train_data[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
cph.fit(train_data, duration_col='Duration Between Purchases', event_col='Churn')
cph.print_summary()


# Check proportional hazards assumption
results = cph.check_assumptions(train_data[['Duration Between Purchases', 'Churn'] + covariates], p_value_threshold=0.05)
print (results)



# Extract Schoenfeld residuals for plotting
schoenfeld_residuals = cph.compute_residuals(train_data, kind='schoenfeld')

# Plot Schoenfeld residuals
def plot_schoenfeld_residuals(residuals, variable, ax=None):
    if ax is None:
        ax = plt.gca()
    
    ax.scatter(residuals.index, residuals[variable], alpha=0.75, s=10, label='Schoenfeld residuals')
    z = np.polyfit(residuals.index, residuals[variable], 1)
    p = np.poly1d(z)
    ax.plot(residuals.index, p(residuals.index), "r--", label='Trend line')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Residuals of {variable}')
    ax.legend()
    ax.set_title(f'Schoenfeld Residuals for {variable}')

fig, axs = plt.subplots(len(cph.params_), figsize=(12, 2 * len(cph.params_)))
for i, var in enumerate(cph.params_.index):
    plot_schoenfeld_residuals(schoenfeld_residuals, var, ax=axs[i])

plt.tight_layout()
plt.show()




'''CPH modelling code for product category'''
# CPH model for product category
cph = CoxPHFitter()
# Fit the model with covariates
cph.fit(train_data[['Duration Between Purchases', 'Churn', 'Product_Category_Clothing','Product_Category_Electronics', 'Product_Category_Home']], duration_col='Duration Between Purchases', event_col='Churn')
cph.print_summary()
# plot the coefficients
cph.plot()
# find HAZARD RATIO (HR)
hazard_ratios = np.exp(cph.summary['coef'])
cph.baseline_hazard_.plot()
plt.title('Baseline Hazard')
print("Baseline hazard:\n", cph.baseline_hazard_)

# Evaluate model performance on the test set
c_index = cph.score(test_data, scoring_method="concordance_index")
print(f'C-Index: {c_index}')

# Clear any existing plots
plt.clf()

values = [
    [0, 0, 0],  # Base
    [1, 0, 0],  # Clothing
    [0, 1, 0],  # Electronics
    [0, 0, 1]   # Home
    
]
# Plot Survival Functions for different categories
fig, ax = plt.subplots(figsize=(10, 6))
cph.plot_partial_effects_on_outcome(covariates=['Product_Category_Electronics', 'Product_Category_Home', 'Product_Category_Clothing'], values=values, ax=ax)
ax.set_title('Survival Function for Product Category')
ax.set_xlabel('Time in Days')
ax.set_ylabel('Survival Probability')
plt.show()



'''------------------------------------LSTM modelling section----------------------------------------------'''


'''LSTM model function baseline'''
import platform
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def forecast_demand(df, product_category, freq, seq_length=30, epochs=20, batch_size=32):
    # Filter data for the specific product category
    df_filtered = df[df['Product Category'] == product_category]
    df_filtered.set_index('Purchase Date', inplace=True)

    # Resample data based on user input
    if freq == 'D':
        sales_data = df_filtered.resample('D').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'W':
        sales_data = df_filtered.resample('W').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'M':
        sales_data = df_filtered.resample('M').agg({'Quantity': 'sum'}).reset_index()
    else:
        print("Wrong. Please enter 'D' for daily, 'W' for weekly, or 'M' for monthly.")
        return

    sales_data['Purchase Date'] = pd.to_datetime(sales_data['Purchase Date'])
    sales_data.set_index('Purchase Date', inplace=True)

   
    # Create a filename
    # Determine the platform and set the file location accordingly
    if platform.system() == 'Windows':
        filelocation = f'C:/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_baseline_{product_category.lower()}.xlsx'
    else:
        filelocation = f'/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_baseline_{product_category.lower()}.xlsx'
    
    # Save to an Excel file if needed
    sales_data.to_excel(filelocation, index=False, sheet_name='Sheet1')
    print(f'Data saved to {filelocation}')
        
    
    
    # Plot the data to visualize the trend
    plt.figure(figsize=(14, 5))
    plt.plot(sales_data, marker='o', color='b')  # Dotted line with circular markers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Demand Trend for {product_category} before preprocessing ({freq})')
    plt.xlabel('Purchase Date')
    plt.ylabel('Quantity')
    plt.show()
    
    sales_data.describe()


    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data)

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], [] 
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM input (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    #train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    #predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reverse scaling

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_unscaled, color='blue', label='Actual Demand')
    plt.plot(predictions, color='red', label='Predicted Demand')
    plt.xlabel(f'Time in ({freq})')
    plt.ylabel('Demand Quantity')
    plt.title(f'Demand Forecasting for {product_category} baseline ({freq})')
    plt.legend()
    plt.show()

    # Evaluate model performance
    mse = mean_squared_error(y_test_unscaled, predictions)
    mae = mean_absolute_error(y_test_unscaled, predictions)
    print(f'Mean Squared Error for {product_category}: {mse}')
    print(f'Mean Absolute Error for {product_category}: {mae}')

# Load the sample dataset
if platform.system() == 'Windows':
    df=pd.read_csv("C:/Users/drayden foo/OneDrive - Sunway Education Group/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv", low_memory=False)
else:
    df=pd.read_csv("/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv")

# change Purchase Date to into datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y %H:%M')

product_categories = ['Electronics', 'Home', 'Books', 'Clothing']

# Prompt the user to enter the frequency
frequency = input("Enter the frequency for data aggregation (D for daily, W for weekly, M for monthly) for (baseline): ").strip().upper()

# Call the function for each product category with the user-specified frequency
for category in product_categories:
    forecast_demand(df, category, frequency)







'''LSTM model function for removing outlier'''

# Define the function
def forecast_demand(df, product_category, freq, seq_length=30, epochs=20, batch_size=32):
    # Filter data for the specific product category
    df_filtered = df[df['Product Category'] == product_category]

    # Set 'Purchase Date' as the index
    df_filtered.set_index('Purchase Date', inplace=True)

    # Resample data based on user input
    if freq == 'D':
        sales_data = df_filtered.resample('D').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'W':
        sales_data = df_filtered.resample('W').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'M':
        sales_data = df_filtered.resample('M').agg({'Quantity': 'sum'}).reset_index()
    else:
        print("Invalid frequency. Please enter 'D' for daily, 'W' for weekly, or 'M' for monthly.")
        return

    # Plot the data to visualize the trend
    plt.figure(figsize=(14, 5))
    plt.plot(sales_data['Purchase Date'], sales_data['Quantity'], marker='o', color='b')  # Dotted line with circular markers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Demand Trend for {product_category} before preprocessing ({freq})')
    plt.xlabel('Purchase Date')
    plt.ylabel('Quantity')
    plt.show()

    print(sales_data.describe())

    # Create a filename
    if platform.system() == 'Windows':
        filelocation = f'C:/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_baseline_{product_category.lower()}.xlsx'
    else:
        filelocation = f'/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_baseline_{product_category.lower()}.xlsx'

    # Save to an Excel file if needed
    sales_data.to_excel(filelocation, index=False, sheet_name='Sheet1')
    print(f'Data saved to {filelocation}')

    # Detect outliers using IQR method
    def detect_outliers_iqr(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

    # Detect outliers in 'Quantity'
    lower_bound, upper_bound = detect_outliers_iqr(sales_data['Quantity'])

    # Filter outliers
    non_outliers = sales_data[(sales_data['Quantity'] >= lower_bound) & (sales_data['Quantity'] <= upper_bound)]

    # Display outliers
    outliers = sales_data[(sales_data['Quantity'] < lower_bound) | (sales_data['Quantity'] > upper_bound)]
    print(f"Outliers for {product_category}:")
    print(outliers)

    # Print the number of outliers removed
    num_outliers_removed = len(sales_data) - len(non_outliers)
    print(f"Number of outliers removed for {product_category}: {num_outliers_removed}")
    print(f"Outliers thresholds: lower bound = {lower_bound}, upper bound = {upper_bound}")

    print(non_outliers.head())

    # Plot the data after removing outliers
    plt.figure(figsize=(14, 5))
    plt.plot(non_outliers['Purchase Date'], non_outliers['Quantity'], marker='o', color='b')
    plt.title(f'Demand Trend for {product_category} after removing outliers ({freq})')
    plt.xlabel('Purchase Date')
    plt.ylabel('Quantity')
    plt.show()

    print(f"Summary stats for {product_category} after removing outliers:\n{non_outliers.describe()}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(non_outliers[['Quantity']])

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM input (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))  # Input layer specifying the input shape
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reverse scaling

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_unscaled, color='blue', label='Actual Demand')
    plt.plot(predictions, color='red', label='Predicted Demand')
    plt.xlabel(f'Time in ({freq})')
    plt.ylabel('Demand Quantity')
    plt.title(f'Demand Forecasting for {product_category} removing outliers ({freq})')
    plt.legend()
    plt.show()

    # Evaluate model performance
    mse = mean_squared_error(y_test_unscaled, predictions)
    mae = mean_absolute_error(y_test_unscaled, predictions)
    print(f'Mean Squared Error for {product_category}: {mse}')
    print(f'Mean Absolute Error for {product_category}: {mae}')

# Determine the platform and set the file location accordingly
if platform.system() == 'Windows':
    df = pd.read_csv("C:/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv", low_memory=False)
else:
    df = pd.read_csv("/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv")

# Change 'Purchase Date' to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y %H:%M')

product_categories = ['Electronics', 'Home', 'Books', 'Clothing']

# Prompt the user to enter the frequency
frequency = input("Enter the frequency for data aggregation (D for daily, W for weekly, M for monthly): ").strip().upper()

# Call the function for each product category
for category in product_categories:
    forecast_demand(df, category, frequency)




#----------------------------------------------------------------------------------------------------------------------------------------
'''LSTM model function for replacing outlier with mean'''

import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

def forecast_demand(df, product_category, freq, seq_length=30, epochs=20, batch_size=32):
    # Filter data for the specific product category
    df_filtered = df[df['Product Category'] == product_category]
    df_filtered.set_index('Purchase Date', inplace=True)

    # Get user's duration request
    if freq == 'D':
        sales_data = df_filtered.resample('D').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'W':
        sales_data = df_filtered.resample('W').agg({'Quantity': 'sum'}).reset_index()
    elif freq == 'M':
        sales_data = df_filtered.resample('M').agg({'Quantity': 'sum'}).reset_index()
    else:
        print("Invalid frequency. Please enter 'D' for daily, 'W' for weekly, or 'M' for monthly.")
        return

    # Create a filename
    if platform.system() == 'Windows':
        filelocation = f'C:/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_replace_with_mean_{product_category.lower()}.xlsx'
    else:
        filelocation = f'/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/{freq}_sales_replace_with_mean_{product_category.lower()}.xlsx'

    # Save to an Excel file if needed
    sales_data.to_excel(filelocation, index=False, sheet_name='Sheet1')
    print(f'Data saved to {filelocation}')


    # Plot the data to visualize the trend
    plt.figure(figsize=(14, 5))
    plt.plot(sales_data['Purchase Date'], sales_data['Quantity'], marker='o', color='b')  # Dotted line with circular markers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Demand Trend for {product_category} before preprocessing ({freq})')
    plt.xlabel('Purchase Date')
    plt.ylabel('Quantity')
    plt.show()

    print(sales_data.describe())

    # Detect outliers using IQR method
    def detect_outliers_iqr(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

    # Detect outliers in the 'Quantity' column
    lower_bound, upper_bound = detect_outliers_iqr(sales_data['Quantity'])

    # Replace outliers with the mean of the non-outliers
    mean_quantity = sales_data[(sales_data['Quantity'] >= lower_bound) & (sales_data['Quantity'] <= upper_bound)]['Quantity'].mean()
    sales_data['Quantity'] = np.where((sales_data['Quantity'] < lower_bound) | (sales_data['Quantity'] > upper_bound), mean_quantity, sales_data['Quantity'])

    # Display outliers
    outliers = sales_data[(sales_data['Quantity'] <= lower_bound) | (sales_data['Quantity'] >= upper_bound)]
    print(f"Outliers for {product_category}:")
    print(outliers)

    # Print the number of outliers replaced
    num_outliers_replaced = len(outliers)
    print(f"Number of outliers replaced for {product_category}: {num_outliers_replaced}")
    print(f"Outliers thresholds: lower bound = {lower_bound}, upper bound = {upper_bound}")

    print(sales_data.head())

    # Plot the data after replacing outliers
    plt.figure(figsize=(14, 5))
    plt.plot(sales_data['Purchase Date'], sales_data['Quantity'], marker='o', color='b')
    plt.title(f'Demand Trend for {product_category} after replacing outliers ({freq})')
    plt.xlabel('Purchase Date')
    plt.ylabel('Quantity')
    plt.show()

    print(f"Summary stats for {product_category} after replacing outliers:\n{sales_data.describe()}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data[['Quantity']])

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM input (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reverse scaling

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_unscaled, color='blue', label='Actual Demand')
    plt.plot(predictions, color='red', label='Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('Demand Quantity')
    plt.title(f'Demand Forecasting for {product_category} with mean replacement ({freq})')
    plt.legend()
    plt.show()

    # Evaluate model performance
    mse = mean_squared_error(y_test_unscaled, predictions)
    mae = mean_absolute_error(y_test_unscaled, predictions)
    print(f'Mean Squared Error for {product_category}: {mse}')
    print(f'Mean Absolute Error for {product_category}: {mae}')

# Load the sample dataset
if platform.system() == 'Windows':
    df=pd.read_csv("C:/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv", low_memory=False)
else:
    df=pd.read_csv("/Users/draydenfoo/Library/CloudStorage/OneDrive-SunwayEducationGroup/Uni/20058798_Foo Wai Loon_Apr24/System Files/ecommerce_customer_data_large.csv")

# Change 'Purchase Date' to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y %H:%M')

product_categories = ['Electronics', 'Home', 'Books', 'Clothing']

# Prompt the user to enter the frequency
frequency = input("Enter the frequency for data aggregation (D for daily, W for weekly, M for monthly): ").strip().upper()

# Call the function for each product category
for category in product_categories:
    forecast_demand(df, category, frequency)
