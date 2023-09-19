import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import KFold

filePath = r'C:/Users/WeiTh/Desktop/Dehliclimate/DTrain.csv'
# Read CSV file using pandas
df = pd.read_csv(filePath)
# Print the result
print(df)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

date = df["date"]
print(date)


def plot(data_columns):
    data = df[data_columns]
    plt.plot(date, data, marker='o', linestyle='-', color='b')
    plt.title('Time Series Line Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Display the plot
    plt.show()


plot("meantemp")

# Standardize the data
scaler = MinMaxScaler()
datatemp = scaler.fit_transform(df["meantemp"])


def create_time_series(data, time_steps):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 30  # Time window size
X_train, y_train = create_time_series(train_data, time_steps)
X_test, y_test = create_time_series(test_data, time_steps)

# Build an RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, activation='relu', input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predicted_values = model.predict(X_test)

# Inverse transform the standardized data
predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test)

# Print the first few predicted values
print("Predicted Values:", predicted_values[:5].flatten())
print("Actual Values:", y_test[:5].flatten())
