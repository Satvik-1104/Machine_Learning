import numpy as np
import pandas as pd
import math

numIterations = 10000
learningRate = 0.01

# Creating the Dataset
np.random.seed(900)
hoursOfSunshine = np.random.randint(1, 12, 30)
iceCreamsSold = np.random.randint(1, 100, 30)
dataset = np.column_stack((hoursOfSunshine, iceCreamsSold))
dataframe = pd.DataFrame(dataset, columns=['HoursOfSunshine', 'IceCreamsSold'])

# Displaying the Head, #rows, #columns and ranges of each feature
print("Head:")
print(dataframe.head())
print()
print(f"Rows: {dataset.shape[0]}")
print(f"Columns: {dataset.shape[1]}")
print()
print(f"range of Hours of Sunshine: {dataframe['HoursOfSunshine'].min()} - {dataframe['HoursOfSunshine'].max()}")
print(f"range of number of IceCreams sold: {dataframe['IceCreamsSold'].min()} - {dataframe['IceCreamsSold'].max()}")
print("======================================================")


def linear_regression(x, y, epochs, alpha):
    theta = np.zeros(x.shape[1] + 1)
    m = x.shape[0]
    x = np.c_[np.ones((m, 1)), x]
    for i in range(epochs):
        predictions = x.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * x.T.dot(errors)
        theta -= alpha * gradients

    return theta


def predict(x, theta):
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x.dot(theta)


def mean_squared_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


# Splitting the Dataset into 70-30,80-20,90-10 training : testing ratios
# And applying simple Linear Regression to predict the number of IceCreams given the hours of Sunshine
ratios = [0.7, 0.8, 0.9]
for ratio in ratios:
    trainRatio = ratio
    trainSize = round(dataset.shape[0] * trainRatio)
    testSize = dataset.shape[0] - trainSize
    indices = np.array([i for i in range(dataset.shape[0])])
    np.random.shuffle(indices)
    trainIndices = indices[:trainSize]
    testIndices = indices[trainSize:]
    trainDf = dataframe.iloc[trainIndices]
    testDf = dataframe.iloc[testIndices]
    x_train = trainDf['HoursOfSunshine'].values.reshape(-1, 1)
    y_train = trainDf['IceCreamsSold'].values
    x_test = testDf['HoursOfSunshine'].values.reshape(-1, 1)
    y_test = testDf['IceCreamsSold'].values
    parameters = linear_regression(x_train, y_train, numIterations, learningRate)
    y_predicted_train = predict(x_train, parameters)
    y_predicted_test = predict(x_test, parameters)
    trainMSE = mean_squared_error(y_train, y_predicted_train)
    testMSE = mean_squared_error(y_test, y_predicted_test)

    print(f"Train : Test ratio - {int(ratio * 100)} : {math.ceil((1 - ratio) * 100)}")
    print(f"Parameters: {parameters}")
    print(f"Train MSE: {trainMSE}")
    print(f"Test MSE: {testMSE}")
    print("-------------------------------------------------------")
