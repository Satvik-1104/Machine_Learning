import numpy as np
import pandas as pd

np.random.seed(9)
num_iterations = [500, 1000, 1500]
learningRates = [0.01, 0.005, 0.001]


# Sigmoid
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


# Hypothesis
def heuristic_function(theta, x):
    return sigmoid_function(x.dot(theta))


# Cost
def cost_function(y, theta, x):
    epsilon = 1e-5
    m = x.shape[0]
    h = heuristic_function(theta, x)
    return -(1 / m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))


# Gradient Descent
def gradient_descent(x_t, y_t, x_v, y_v, epochs, alpha, org_parameters):
    theta = np.copy(org_parameters)
    m = x_t.shape[0]
    for i in range(epochs):
        h = heuristic_function(theta, x_t)
        errors = h - y_t.flatten()
        gradients = (1 / m) * x_t.T.dot(errors)
        theta -= alpha * gradients
    validation_mse = cost_function(y_v, theta, x_v)
    return theta, validation_mse


def predict(x_t, theta):
    return heuristic_function(theta, x_t) >= 0.5


# Confusion Matrix
def confusion_matrix(y_true, y_predicted):
    y_predicted = y_predicted.astype(int)
    tp = np.sum((y_true == 1) & (y_predicted == 1))
    fp = np.sum((y_true == 0) & (y_predicted == 1))
    fn = np.sum((y_true == 1) & (y_predicted == 0))
    tn = np.sum((y_true == 0) & (y_predicted == 0))
    return np.array([[tp, fp], [fn, tn]])


# Dataset
num_samples = 1000
age = np.random.randint(18, 62, size=num_samples)
monthlyCharges = np.random.uniform(1000, 10000, size=num_samples)
contractType = np.random.choice([0, 1], size=num_samples)
tenure = np.random.randint(1, 72, size=num_samples)
churn = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

dataframe = pd.DataFrame({
    'Age': age,
    'Monthly_Charges': monthlyCharges,
    'Contract_Type': contractType,
    'Tenure': tenure,
    'Churn': churn
})


# Normalise
def normalise_features(feature):
    return (feature - np.mean(feature)) / np.std(feature)


def accuracy(c_matrix):
    return c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1])


def recall(c_matrix):
    return c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0])


dataframe['Age'] = normalise_features(dataframe['Age'])
dataframe['Monthly_Charges'] = normalise_features(dataframe['Monthly_Charges'])
dataframe['Tenure'] = normalise_features(dataframe['Tenure'])
print(dataframe.head())

num_rows = dataframe.shape[0]
# Append 1's
dataframe.insert(0, 'Bias', 1)

trainRatio = 0.6
validationRatio = 0.2
testRatio = 0.2
trainSize = int(num_rows * trainRatio)
validationSize = int(num_rows * validationRatio)
indices = np.array([i for i in range(num_samples)])
np.random.shuffle(indices)
trainIndices = indices[:trainSize]
validationIndices = indices[trainSize:trainSize + validationSize]
testIndices = indices[trainSize + validationSize:]
trainDf = dataframe.iloc[trainIndices]
validationDf = dataframe.iloc[validationIndices]
testDf = dataframe.iloc[testIndices]
x_train = trainDf[['Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_train = trainDf['Churn'].values.reshape(-1, 1)
x_validation = validationDf[['Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_validation = validationDf['Churn'].values.reshape(-1, 1)
x_test = testDf[['Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_test = testDf['Churn'].values.reshape(-1, 1)

num_cols = x_train.shape[1]
parameters = np.ones(num_cols)
bestVMSE = float('inf')
bestIterations = 0
bestLearningRate = 0
bestParameters = None
for itr in num_iterations:
    for learningRate in learningRates:
        parameters, vMSE = gradient_descent(x_train, y_train, x_validation, y_validation, itr, learningRate, parameters)
        if vMSE < bestVMSE:
            # Choosing the Best
            bestVMSE = vMSE
            bestIterations = itr
            bestLearningRate = learningRate
            bestParameters = parameters

print(f"Best Parameters: {bestParameters}")
print(f"Best Learning Rate: {bestLearningRate}")
print(f"Best Number of Iterations: {bestIterations}")
print(f"Best Validation MSE: {bestVMSE}")

test_prediction = predict(x_test, bestParameters)
testMSE = cost_function(y_test, bestParameters, x_test)
print(f"Test MSE: {testMSE}")

churn_test = y_test[:, -1]
cnfMatrix = confusion_matrix(churn_test, test_prediction)
print(cnfMatrix)

acc = accuracy(cnfMatrix) * 100
print(f"Accuracy: {acc}%")

rec = recall(cnfMatrix) * 100
print(f"Recall: {rec}%")
