import numpy as np
import pandas as pd

X = pd.DataFrame(np.array([[1500, 3, 20, 10],
                           [1800, 4, 15, 8],
                           [2000, 4, 5, 12],
                           [1200, 2, 30, 15],
                           [1700, 3, 10, 7],
                           [2500, 5, 8, 5],
                           [1300, 2, 25, 18],
                           [1900, 4, 12, 16],
                           [1600, 3, 20, 9],
                           [2100, 4, 5, 11]]),
                 columns=['Size', 'Bedrooms', 'Age', 'Distance'])

y = pd.Series(np.array([300, 400, 500, 200, 350, 600, 220, 320, 330, 480]))

data = pd.concat([X, y], axis=1)

shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.7 * len(shuffled_data))
val_size = int(0.15 * len(shuffled_data))

train_data = shuffled_data[:train_size]
val_data = shuffled_data[train_size:train_size + val_size]
test_data = shuffled_data[train_size + val_size:]

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]


def normalize(df):
    min_vals = df.min()
    max_vals = df.max()
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    return (df - min_vals) / ranges


min_vals = X_train.min()
ranges = X_train.max() - min_vals
ranges[ranges == 0] = 1

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


def train_linear_regression(X, y, learning_rate, num_iterations):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Add intercept term
    theta = np.zeros(n + 1)

    for i in range(num_iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradients = np.dot(X.T, errors) / m
        theta -= learning_rate * gradients

    return theta


def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
    return np.dot(X, theta)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


learning_rates = [0.001, 0.01, 0.1, 0.5]
epochs = [500, 1000]
best_lr = None
best_epoch = None
best_val_mse = float('inf')

for lr in learning_rates:
    for epoch in epochs:
        theta = train_linear_regression(X_train.values, y_train.values, learning_rate=lr, num_iterations=epoch)

        # Validation set prediction and error
        y_val_pred = predict(X_val.values, theta)
        val_mse = mean_squared_error(y_val.values, y_val_pred)

        print(f"Learning Rate: {lr}, Epochs: {epoch}, Validation MSE: {val_mse}")

        # Keep track of the best learning rate
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_lr = lr

print(f"Best Learning Rate: {best_lr}, Best epoch: {best_epoch}")

best_theta = train_linear_regression(X_train.values, y_train.values, learning_rate=best_lr, num_iterations=1000)

y_test_pred = predict(X_test.values, best_theta)
test_mse = mean_squared_error(y_test.values, y_test_pred)

print(f"Test Set MSE: {test_mse}")


def predict_house_price(size, bedrooms, age, distance, min_vals, ranges, theta):
    # Create DataFrame with the house details
    new_data = pd.DataFrame([[size, bedrooms, age, distance]], columns=['Size', 'Bedrooms', 'Age', 'Distance'])

    new_data_normalized = (new_data - min_vals) / ranges
    new_data_normalized = new_data_normalized.fillna(0)  # Replace NaN values with 0 if any

    price = predict(new_data_normalized.values, theta)
    return price[0]


size = 2500
bedrooms = 4
age = 10
distance = 5

predicted_price = predict_house_price(size, bedrooms, age, distance, min_vals, ranges, best_theta)
print(f"Predicted Price: ${predicted_price * 1000:.2f}")
