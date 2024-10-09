import numpy as np
import pandas as pd

np.random.seed(9)


def normalise_features(feature):
    return (feature - np.mean(feature)) / np.std(feature)


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


def hypothesis_function(theta, x):
    argument = x.dot(theta.T)
    return sigmoid_function(argument)


def cost_function(x, y, theta):
    m = x.shape[0]
    epsilon = 1e-5
    h = hypothesis_function(theta, x)
    return -(1 / m) * (y.T.dot(np.log(h + epsilon)) + (1-y).T.dot(np.log(1 - h - epsilon)))


def gradient_descent(x_t, y_t, x_v, y_v, epochs, alpha, org_parameters):
    m = x_t.shape[0]
    theta = np.copy(org_parameters).reshape(-1, 1)
    for i in range(epochs):
        predictions = sigmoid_function(x_t.dot(theta))
        errors = predictions - y_t
        gradients = (1 / m) * x_t.T.dot(errors)
        theta -= alpha * gradients
    v_mse = cost_function(x_v, y_v, theta.T)
    return theta.T, v_mse


def one_vs_all(x_t, y_t, x_v, y_v, num_classes, epochs, alpha, org_parameters):
    n = x_t.shape[1]
    vmse_total = 0
    all_theta = np.copy(org_parameters)
    for i in range(num_classes):
        y_t_i = np.where(y_t == i, 1, 0)
        y_v_i = np.where(y_v == i, 1, 0)
        theta_i = all_theta[i].reshape(-1, 1)
        theta, vmse = gradient_descent(x_t, y_t_i.reshape(-1, 1), x_v, y_v_i.reshape(-1, 1), epochs, alpha, theta_i)
        vmse_total += vmse
        all_theta[i] = theta.ravel()
    return all_theta, vmse_total


def predict_for_all_theta(x_t, all_theta):
    predictions = sigmoid_function(x_t.dot(all_theta.T))
    return np.argmax(predictions, axis=1)


def build_confusion_matrix(y_true, y_predicted, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_predicted[i], y_true[i]] += 1
    return conf_matrix


def calculate_accuracy(conf_matrix, num_classes):
    total = 0
    correct = 0
    for i in range(num_classes):
        for j in range(num_classes):
            total += conf_matrix[i][j]
            if i == j:
                correct += conf_matrix[i][j]
    return (correct / total) * 100


num_samples = 1000
age = np.random.randint(18, 62, size=num_samples)
monthly_charges = np.random.uniform(1000, 10000, size=num_samples)
contract_type = np.random.choice([0, 1], size=num_samples)
tenure = np.random.randint(0, 72, size=num_samples)
churn = np.random.choice([0, 1, 2], size=num_samples, p=[0.6, 0.2, 0.2])

dataframe = pd.DataFrame(
    {
        'Age': age,
        'Monthly_Charges': monthly_charges,
        'Contract_Type': contract_type,
        'Tenure': tenure,
        'Churn': churn
    }
)

dataframe['Age'] = normalise_features(dataframe['Age'])
dataframe['Monthly_Charges'] = normalise_features(dataframe['Monthly_Charges'])
"""dataframe['Contract_Type'] = normalise_features(dataframe['Contract_Type'])"""
dataframe['Tenure'] = normalise_features(dataframe['Tenure'])

dataframe.insert(0, 'Bias', 1)

print(dataframe.head(10))

num_rows = dataframe.shape[0]
trainRatio = 0.6
validationRatio = 0.2
testRatio = 0.2
trainSize = int(trainRatio * num_rows)
validationSize = int(validationRatio * num_rows)
indices = np.array([i for i in range(num_samples)])
np.random.shuffle(indices)
trainIndices = indices[:trainSize]
validationIndices = indices[trainSize:trainSize + validationSize]
testIndices = indices[trainSize + validationSize:]
trainDf = dataframe.iloc[trainIndices]
validationDf = dataframe.iloc[validationIndices]
testDf = dataframe.iloc[testIndices]
x_train = trainDf[['Bias', 'Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_train = trainDf['Churn'].values.reshape(-1, 1)
x_validation = validationDf[['Bias', 'Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_validation = validationDf['Churn'].values.reshape(-1, 1)
x_test = testDf[['Bias', 'Age', 'Monthly_Charges', 'Contract_Type', 'Tenure']].values
y_test = testDf['Churn'].values.reshape(-1, 1)

num_iterations = [1000, 1500, 2000]
learningRates = [0.01, 0.05, 0.001]
parameters = np.ones((3, x_train.shape[1]))
bestParameters = None
bestLearningRate = 0
bestIterations = 0
bestMSE = float('inf')
for num_itr in num_iterations:
    for learningRate in learningRates:
        parameters, vMSE = one_vs_all(x_train, y_train, x_validation, y_validation, 3, num_itr, learningRate, parameters)
        if vMSE < bestMSE:
            bestParameters = parameters
            bestIterations = num_itr
            bestLearningRate = learningRate
            bestMSE = vMSE

print(f"Best Parameters: {bestParameters}")
print(f"Best Learning Rate: {bestLearningRate}")
print(f"Best Number of Iterations: {bestIterations}")
print(f"Best Validation MSE: {bestMSE}")

testPredictions = predict_for_all_theta(x_test, bestParameters)
conf_matrix = build_confusion_matrix(y_test, testPredictions, 3)
accuracy = calculate_accuracy(conf_matrix, 3)

print("Confusion Matrix")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
