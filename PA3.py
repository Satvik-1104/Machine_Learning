import numpy as np
import pandas as pd
import math

np.random.seed(0)
hoursOfStudy = [1, 2, 3, 4, 5]
examScore = [2, 4, 6, 8, 10]
trainRatio = 0.8
learningRate = 0.01
numIterations = 1000


def batch_gradient_descent(x, y, epochs, alpha):
    m = x.shape[0]
    x = np.c_[np.ones((m, 1)), x]
    theta = np.ones(x.shape[1])
    for i in range(epochs):
        predictions = x.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * x.T.dot(errors)
        theta -= alpha * gradients
    return theta


def stochastic_gradient_descent(x, y, epochs, alpha):
    m = x.shape[0]
    choices = [i for i in range(m)]
    x = np.c_[np.ones((m, 1)), x]
    theta = np.ones(x.shape[1])
    for i in range(epochs):
        index = np.random.choice(choices)
        prediction = x[index].dot(theta)
        error = prediction - y[index]
        gradient = x[index].T.dot(error)
        theta -= alpha * gradient
    return theta


def predict_values(x, theta):
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x.dot(theta)


def mean_squared_error(y_true, y_predicted):
    return np.mean((y_predicted - y_true) ** 2)


dataset = np.column_stack((hoursOfStudy, examScore))
dataframe = pd.DataFrame(dataset, columns=['hoursOfStudy', 'examScore'])
num_rows = dataset.shape[0]
num_cols = dataset.shape[1]
indices = np.array([i for i in range(num_rows)])
np.random.shuffle(indices)
trainSize = int(num_rows * trainRatio)
trainIndices = indices[:trainSize]
testIndices = indices[trainSize:]
trainDf = dataframe.iloc[trainIndices]
testDf = dataframe.iloc[testIndices]
x_train = trainDf['hoursOfStudy'].values.reshape(-1, 1)
y_train = trainDf['examScore'].values
x_test = testDf['hoursOfStudy'].values.reshape(-1, 1)
y_test = testDf['examScore'].values
print("======================================================")
print("Batch Gradient Descent")
print("------------------------------------------------------")
parametersBGD = batch_gradient_descent(x_train, y_train, numIterations, learningRate)
train_pred_BGD = predict_values(x_train, parametersBGD)
test_pred_BGD = predict_values(x_test, parametersBGD)
train_mse_BGD = mean_squared_error(y_train, train_pred_BGD)
test_mse_BGD = mean_squared_error(y_test, test_pred_BGD)
print(f"train-test ratio: {int(trainRatio * 100)} : {math.ceil((1 - trainRatio) * 100)}")
print(f"Parameters (BGD): {parametersBGD}")
print(f"Train MSE (BGD): {train_mse_BGD}")
print(f"Test MSE (BGD): {test_mse_BGD}")
print(f"Test x and predicted value (BGD): \n{testDf}")
print()
print("======================================================")
print("Stochastic Gradient Descent")
print("------------------------------------------------------")
parametersSGD = stochastic_gradient_descent(x_train, y_train, numIterations, learningRate)
train_pred_SGD = predict_values(x_train, parametersSGD)
test_pred_SGD = predict_values(x_test, parametersSGD)
train_mse_SGD = mean_squared_error(y_train, train_pred_SGD)
test_mse_SGD = mean_squared_error(y_test, test_pred_SGD)
print(f"train-test ratio: {int(trainRatio * 100)} : {math.ceil((1 - trainRatio) * 100)}")
print(f"Parameters (SGD): {parametersSGD}")
print(f"Train MSE (SGD): {train_mse_SGD}")
print(f"Test MSE (SGD): {test_mse_SGD}")
print(f"Test x and predicted value (SGD): \n{testDf}")
print()
