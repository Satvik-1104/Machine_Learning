import numpy as np
import pandas as pd

np.random.seed(9)


def linear_regression(x_t, y_t, x_v, y_v, epochs, alpha, org_theta):
    m = x_t.shape[0]
    x_t = np.c_[np.ones((m, 1)), x_t]
    theta = np.copy(org_theta)
    for i in range(epochs):
        predictions = x_t.dot(theta)
        errors = y_t - predictions
        gradients = (1 / m) * x_t.T.dot(errors)
        theta -= gradients * alpha
    predicted_values = predict_values(x_v, theta)
    v_mse = mean_squared_error(y_v, predicted_values)
    return theta, v_mse


def predict_values(x, theta):
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x.dot(theta)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


hoursOfStudy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
examScore = [2, 4, 6, 8, 10, 12, 14, 15, 16, 20]
dataset = np.column_stack((hoursOfStudy, examScore))
dataframe = pd.DataFrame(dataset, columns=['hoursOfStudy', 'examScore'])

# IMP - Normalization Step
dataframe['hoursOfStudy'] = (dataframe['hoursOfStudy'] - dataframe['hoursOfStudy'].mean()) / dataframe['hoursOfStudy'].std()

trainRatio = 0.6
validationRatio = 0.2
testRatio = 0.2
datasetLength = dataset.shape[0]
trainSize = int(trainRatio * datasetLength)
validationSize = int(validationRatio * datasetLength)

indices = np.array([i for i in range(datasetLength)])
np.random.shuffle(indices)
trainIndices = indices[:trainSize]
validationIndices = indices[trainSize:trainSize + validationSize]
testIndices = indices[trainSize + validationSize:]

trainDf = dataframe.iloc[trainIndices]
testDf = dataframe.iloc[testIndices]
validationDf = dataframe.iloc[validationIndices]

x_train = trainDf['hoursOfStudy'].values.reshape(-1, 1)
x_validation = validationDf['hoursOfStudy'].values.reshape(-1, 1)
x_test = testDf['hoursOfStudy'].values.reshape(-1, 1)
y_train = trainDf['examScore'].values
y_validation = validationDf['examScore'].values
y_test = testDf['examScore'].values

numIterations = [250, 500, 1000]
learningRates = [0.01, 0.005, 0.001]
best_MSE = float('inf')
best_learningRate = 0
best_numIterations = 0
best_parameters = None

parameters = np.ones(x_test.shape[1] + 1)

for itr in numIterations:
    for learningRate in learningRates:
        new_parameters, validation_mse = linear_regression(x_train, y_train, x_validation, y_validation, itr, learningRate, parameters)
        print(f"Iterations: {itr}, Learning Rate: {learningRate}, Validation MSE: {validation_mse}")
        if validation_mse < best_MSE:
            best_MSE = validation_mse
            best_parameters = new_parameters
            best_learningRate = learningRate
            best_numIterations = itr

print(f"Best Parameters: {best_parameters}")
print(f"Best Learning Rate: {best_learningRate}")
print(f"Best Number of Iterations: {best_numIterations}")
print(f"Best Validation MSE: {best_MSE}")

test_predictions = predict_values(x_test, parameters)
test_MSE = mean_squared_error(y_test, test_predictions)
print(f"Test MSE: {test_MSE}")
